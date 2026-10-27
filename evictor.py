#!/usr/bin/env python3
"""
evictor.py — size-bounded eviction for vLLM's KV-cache offload secondary tiers:
the filesystem tier and the object-store (S3-compatible) tier.

Context: vLLM's multi-tier KV offloading (vllm-project/vllm#38260) lets vLLM
spill KV cache blocks from GPU memory down to CPU memory, then to a filesystem
or S3-compatible object-store tier, with each block stored content-addressed.
Eviction for both secondary tiers is explicitly out of scope for vLLM itself —
a vLLM maintainer wrote on that issue that "eviction should be managed outside
of vLLM," and a real user's request for a disk-usage limit on the fs tier went
unanswered there. This is that external controller.

Known limitation, stated up front rather than hidden: vLLM does not emit a
read/hit event for secondary-tier blocks, so there is no exact way to know
which blocks are "hot" from outside the process.

- Filesystem tier: the only available recency signal is the filesystem's
  atime on each block file, updated by vLLM's own read when it serves a cache
  hit. That only works if the filesystem actually updates atime on read — not
  mounted noatime, and not relatime-throttled to the point of being stale.
  FilesystemBackend.detect_recency_mode() checks this empirically at startup;
  if it fails, eviction falls back to mtime (oldest-written wins, not true
  LRU) with a loud warning rather than silently claiming LRU it can't deliver.

- Object-store tier: S3 has no atime equivalent at all. The only signal
  available through the plain object API is LastModified (write time), so
  S3Backend always runs oldest-written-first, never true LRU, and says so
  loudly every run. True LRU there would need S3 server access logs or
  CloudTrail data events correlated back to object keys — out of scope here.

Second known limitation, both tiers: eviction is not coordinated with vLLM's
own reads. A block could be mid-read when this tool deletes it. The min-age
grace period is a mitigation, not a guarantee — see README.md.
"""
import argparse
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("vllm_kv_evictor")


@dataclass
class Block:
    path: Path  # backend-specific identifier: a filesystem Path, or an S3 object key (str)
    size: int
    atime: float
    mtime: float


# ─────────────────────────────────────────────
# eviction selection — backend-agnostic, works on plain Block objects
# ─────────────────────────────────────────────
def plan_eviction(blocks, capacity_bytes, low_water_bytes, min_age_seconds, now, recency="atime"):
    """The subset of `blocks` to evict, oldest-recency-first, stopping once the
    projected total size is at or below low_water_bytes. Never selects a block
    younger than min_age_seconds by the chosen recency signal — that grace period
    is the main defense against evicting a block mid-read."""
    total = sum(b.size for b in blocks)
    if total <= capacity_bytes:
        return []
    key = (lambda b: b.atime) if recency == "atime" else (lambda b: b.mtime)
    eligible = [b for b in blocks if now - key(b) >= min_age_seconds]
    eligible.sort(key=key)
    to_evict = []
    freed = 0
    for b in eligible:
        if total - freed <= low_water_bytes:
            break
        to_evict.append(b)
        freed += b.size
    return to_evict


# ─────────────────────────────────────────────
# filesystem backend — vLLM's fs secondary tier (root_dir/_r<rank>/)
# ─────────────────────────────────────────────
def iter_rank_dirs(root_dir: Path):
    if not root_dir.is_dir():
        return
    for p in sorted(root_dir.iterdir()):
        if p.is_dir() and p.name.startswith("_r"):
            yield p


def scan_blocks(root_dir: Path):
    """Every block file under root_dir/_r*/, recursively (vLLM may nest further
    inside a rank directory)."""
    blocks = []
    for rank_dir in iter_rank_dirs(root_dir):
        for dirpath, _, filenames in os.walk(rank_dir):
            for name in filenames:
                p = Path(dirpath) / name
                try:
                    st = p.stat()
                except OSError:
                    continue  # raced with a delete/write elsewhere; skip it this pass
                blocks.append(Block(path=p, size=st.st_size, atime=st.st_atime, mtime=st.st_mtime))
    return blocks


def detect_atime_support(root_dir: Path) -> bool:
    """Writes a probe file, back-dates its atime, reads it, and checks whether
    atime actually moved. Catches noatime mounts outright; the back-dating also
    defeats relatime's "only update if the current atime predates mtime, or is
    over a day old" rule, so a relatime mount (the common Linux default) still
    reports correctly instead of being mistaken for noatime."""
    root_dir.mkdir(parents=True, exist_ok=True)
    probe = root_dir / ".evictor_atime_probe"
    try:
        probe.write_bytes(b"x")
        old = time.time() - 2 * 86400
        os.utime(probe, (old, old))
        before = probe.stat().st_atime
        time.sleep(0.01)
        probe.read_bytes()
        after = probe.stat().st_atime
        return after > before and after > old
    except OSError as e:
        log.warning("atime probe failed, assuming unsupported: %s", e)
        return False
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def evict(blocks, dry_run=False):
    freed = 0
    errors = []
    for b in blocks:
        if dry_run:
            freed += b.size
            continue
        try:
            b.path.unlink()
            freed += b.size
        except OSError as e:
            errors.append({"path": str(b.path), "error": str(e)})
    return freed, errors


class FilesystemBackend:
    """vLLM's fs secondary tier: root_dir/_r<rank>/<content-addressed file>."""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)

    def scan(self):
        return scan_blocks(self.root_dir)

    def detect_recency_mode(self) -> str:
        return "atime" if detect_atime_support(self.root_dir) else "mtime"

    def delete(self, blocks, dry_run=False):
        return evict(blocks, dry_run=dry_run)


# ─────────────────────────────────────────────
# S3 (object-store) backend — vLLM's object-store secondary tier, via NIXL
# ─────────────────────────────────────────────
class S3Backend:
    """vLLM's object-store secondary tier: an S3-compatible bucket/prefix of
    content-addressed objects. boto3 is imported lazily so fs-only users never
    need it installed."""

    def __init__(self, bucket, prefix="", endpoint_url=None, region_name=None):
        import boto3  # local import: optional dependency, only needed for this backend

        self.bucket = bucket
        self.prefix = prefix
        self.client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region_name)

    def scan(self):
        blocks = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                ts = obj["LastModified"].timestamp()
                blocks.append(Block(path=obj["Key"], size=obj["Size"], atime=ts, mtime=ts))
        return blocks

    def detect_recency_mode(self) -> str:
        # S3 has no read-time signal through the plain object API at all -- always
        # oldest-written-first, never true LRU, regardless of any probe.
        return "mtime"

    def delete(self, blocks, dry_run=False):
        if dry_run:
            return sum(b.size for b in blocks), []
        freed = 0
        errors = []
        by_key = {b.path: b for b in blocks}
        keys = list(by_key)
        for i in range(0, len(keys), 1000):  # delete_objects caps at 1000 keys/request
            chunk = keys[i:i + 1000]
            resp = self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": k} for k in chunk]},
            )
            for d in resp.get("Deleted", []):
                freed += by_key[d["Key"]].size
            for err in resp.get("Errors", []):
                errors.append({"path": err["Key"], "error": err.get("Message", "unknown")})
        return freed, errors


# ─────────────────────────────────────────────
# backend-agnostic run + the filesystem convenience wrapper kept for compatibility
# ─────────────────────────────────────────────
def run_once_backend(backend, capacity_bytes, low_water_bytes, min_age_seconds, recency, dry_run):
    blocks = backend.scan()
    total_before = sum(b.size for b in blocks)
    to_evict = plan_eviction(blocks, capacity_bytes, low_water_bytes, min_age_seconds, time.time(), recency)
    freed, errors = backend.delete(to_evict, dry_run=dry_run)
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scanned_files": len(blocks),
        "total_bytes_before": total_before,
        "capacity_bytes": capacity_bytes,
        "low_water_bytes": low_water_bytes,
        "recency": recency,
        "evicted_files": len(to_evict),
        "bytes_freed": freed,
        "dry_run": dry_run,
        "errors": errors,
    }


def run_once(root_dir, capacity_bytes, low_water_bytes, min_age_seconds, recency, dry_run):
    """Filesystem convenience wrapper. For S3 or any other backend, build the
    Backend instance directly and call run_once_backend()."""
    return run_once_backend(FilesystemBackend(root_dir), capacity_bytes, low_water_bytes,
                             min_age_seconds, recency, dry_run)


_SIZE_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)?$", re.IGNORECASE)
_SIZE_MULT = {
    "B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4,
    "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4,
}


def parse_size(s: str) -> int:
    m = _SIZE_RE.match(s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"not a size: {s!r} (e.g. 500GB, 2TiB, 1048576)")
    value, unit = m.groups()
    return int(float(value) * _SIZE_MULT[(unit or "B").upper()])


def build_parser():
    p = argparse.ArgumentParser(
        prog="vllm-kv-evictor",
        description="Size-bounded eviction for vLLM's KV-cache offload secondary tiers.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--capacity", type=parse_size, required=True,
                         help="evict once total size exceeds this, e.g. 500GB")
    common.add_argument("--low-water", type=parse_size, default=None,
                         help="evict down to this size (default: 90%% of --capacity)")
    common.add_argument("--min-age", type=float, default=60.0,
                         help="never evict a block accessed/written more recently than this many "
                              "seconds ago (default 60)")
    common.add_argument("--poll-interval", type=float, default=30.0,
                         help="seconds between scans in loop mode (default 30)")
    common.add_argument("--once", action="store_true", help="run a single pass and exit (default: loop)")
    common.add_argument("--dry-run", action="store_true", help="report what would be evicted; delete nothing")
    common.add_argument("--json-log", action="store_true", help="emit one JSON object per pass to stdout")
    common.add_argument("--log-level", default="INFO")

    sub = p.add_subparsers(dest="backend_kind", required=True)

    fs_p = sub.add_parser("fs", parents=[common], help="vLLM's filesystem offload tier")
    fs_p.add_argument("root_dir", type=Path, help="vLLM's fs-tier root_dir (same path given to vLLM)")

    s3_p = sub.add_parser("s3", parents=[common], help="vLLM's object-store offload tier (S3-compatible, via NIXL)")
    s3_p.add_argument("bucket", help="bucket name")
    s3_p.add_argument("--prefix", default="", help="only evict under this key prefix")
    s3_p.add_argument("--endpoint-url", default=None, help="for S3-compatible non-AWS stores")
    s3_p.add_argument("--region", default=None)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    low_water = args.low_water if args.low_water is not None else int(args.capacity * 0.9)
    if low_water > args.capacity:
        log.error("--low-water (%d) must not exceed --capacity (%d)", low_water, args.capacity)
        return 2

    if args.backend_kind == "fs":
        backend = FilesystemBackend(args.root_dir)
        recency = backend.detect_recency_mode()
        if recency == "mtime":
            log.warning(
                "atime does not appear to update on this filesystem (noatime mount, or "
                "unsupported) -- falling back to mtime. Eviction will be oldest-written-"
                "first, NOT least-recently-used: a block read often but written long ago "
                "will be evicted ahead of a block never read since it was written."
            )
    else:
        backend = S3Backend(args.bucket, prefix=args.prefix, endpoint_url=args.endpoint_url,
                             region_name=args.region)
        recency = backend.detect_recency_mode()
        log.warning(
            "S3 has no read-time signal available through the object API -- eviction is "
            "oldest-written-first (by LastModified), NOT least-recently-used. A block read "
            "often but written long ago will be evicted ahead of one written recently but "
            "never read. True LRU here needs S3 access logs or CloudTrail data events, which "
            "this tool does not use."
        )

    stop = {"flag": False}

    def _handle_signal(signum, frame):
        log.info("received signal %s, stopping after this pass", signum)
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while True:
        result = run_once_backend(backend, args.capacity, low_water, args.min_age, recency, args.dry_run)
        if args.json_log:
            print(json.dumps(result))
        else:
            log.info(
                "scanned=%d total_before=%d evicted=%d freed=%d dry_run=%s",
                result["scanned_files"], result["total_bytes_before"],
                result["evicted_files"], result["bytes_freed"], result["dry_run"],
            )
            for err in result["errors"]:
                log.warning("eviction error: %s: %s", err["path"], err["error"])
        if args.once or stop["flag"]:
            return 0
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
