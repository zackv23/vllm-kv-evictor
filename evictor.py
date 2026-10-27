#!/usr/bin/env python3
"""
evictor.py — size-bounded eviction for vLLM's filesystem KV-cache offload tier.

Context: vLLM's multi-tier KV offloading (vllm-project/vllm#38260) stores each
offloaded KV block as a content-addressed file under root_dir/_r<rank>/. Eviction
for this tier is explicitly out of scope for vLLM itself — a vLLM maintainer wrote
on that issue that "eviction should be managed outside of vLLM," and a real user's
request for a disk-usage limit on the fs tier went unanswered there. This is that
external controller.

Known limitation, stated up front rather than hidden: vLLM does not emit a
read/hit event for secondary-tier blocks, so there is no exact way to know which
blocks are "hot" from outside the process. The only available recency signal is
the filesystem's atime on each block file, updated by vLLM's own read when it
serves a cache hit from this tier. That only works if the filesystem actually
updates atime on read — not mounted noatime, and not relatime-throttled to the
point of being stale for this purpose. detect_atime_support() checks this
empirically at startup; if it fails, the tool falls back to mtime (oldest-written
wins, not true LRU) with a loud warning rather than silently claiming LRU it
can't deliver.

Second known limitation: eviction here is not coordinated with vLLM's own reads.
A block could be mid-read when this tool deletes it. The min-age grace period is
the mitigation, not a guarantee — see README.md.
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
    path: Path
    size: int
    atime: float
    mtime: float


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


def run_once(root_dir, capacity_bytes, low_water_bytes, min_age_seconds, recency, dry_run):
    blocks = scan_blocks(root_dir)
    total_before = sum(b.size for b in blocks)
    to_evict = plan_eviction(blocks, capacity_bytes, low_water_bytes, min_age_seconds, time.time(), recency)
    freed, errors = evict(to_evict, dry_run=dry_run)
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
        description="Size-bounded eviction for vLLM's filesystem KV-cache offload tier.",
    )
    p.add_argument("root_dir", type=Path, help="vLLM's fs-tier root_dir (same path given to vLLM)")
    p.add_argument("--capacity", type=parse_size, required=True,
                    help="evict once total size exceeds this, e.g. 500GB")
    p.add_argument("--low-water", type=parse_size, default=None,
                    help="evict down to this size (default: 90%% of --capacity)")
    p.add_argument("--min-age", type=float, default=60.0,
                    help="never evict a block accessed/written more recently than this many seconds ago (default 60)")
    p.add_argument("--poll-interval", type=float, default=30.0,
                    help="seconds between scans in --loop mode (default 30)")
    p.add_argument("--once", action="store_true", help="run a single pass and exit (default: loop)")
    p.add_argument("--dry-run", action="store_true", help="report what would be evicted; delete nothing")
    p.add_argument("--json-log", action="store_true", help="emit one JSON object per pass to stdout")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    low_water = args.low_water if args.low_water is not None else int(args.capacity * 0.9)
    if low_water > args.capacity:
        log.error("--low-water (%d) must not exceed --capacity (%d)", low_water, args.capacity)
        return 2

    recency = "atime" if detect_atime_support(args.root_dir) else "mtime"
    if recency == "mtime":
        log.warning(
            "atime does not appear to update on this filesystem (noatime mount, or "
            "unsupported) -- falling back to mtime. Eviction will be oldest-written-"
            "first, NOT least-recently-used: a block read often but written long ago "
            "will be evicted ahead of a block never read since it was written."
        )

    stop = {"flag": False}

    def _handle_signal(signum, frame):
        log.info("received signal %s, stopping after this pass", signum)
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while True:
        result = run_once(args.root_dir, args.capacity, low_water, args.min_age, recency, args.dry_run)
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
