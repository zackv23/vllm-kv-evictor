# vllm-kv-evictor

Size-bounded eviction for vLLM's filesystem KV-cache offload tier.

## Why this exists

vLLM's multi-tier KV offloading ([vllm-project/vllm#38260](https://github.com/vllm-project/vllm/issues/38260), shipped as part of vLLM 0.26.0) lets vLLM spill KV cache blocks from GPU memory down through CPU memory to a filesystem or object-store tier. The filesystem tier stores each block as a content-addressed file under `root_dir/_r<rank>/`.

Eviction for that tier is explicitly out of scope for vLLM itself. From the design discussion on that issue:

> "eviction should be managed outside of vLLM" — orozery (vLLM maintainer)

> "Today secondary eviction is internal to the secondary tier" / "Secondary-tier removal/invalidation... remain out of scope." — Change72

A real user asked, on that same issue, whether a disk usage limit could be added for the `fs` tier. As of this writing that question has no answer. This tool is that external controller.

## What it does

- Scans `root_dir/_r*/` for block files.
- When total size exceeds `--capacity`, evicts the least-recently-used blocks until total size is at or below `--low-water` (default: 90% of capacity).
- Never evicts anything younger than `--min-age` seconds — the main defense against evicting a block mid-read, since this runs out-of-process from vLLM with no coordination.
- `--dry-run` reports what it would evict without deleting anything. Use this first.

## Known limitations — read before running against production data

**No read-event stream.** vLLM does not emit an event when a secondary-tier block is read/hit. The only available recency signal from outside the process is the filesystem's `atime` on each block file. This tool probes at startup whether `atime` actually updates on your mount (handles both `noatime`, which disables it outright, and `relatime`, the common Linux default, which only updates atime under specific conditions the probe accounts for). If it can't confirm atime is live, it falls back to `mtime` with a loud warning — that is **not** true LRU, just oldest-written-first, and will evict a frequently-read-but-rarely-rewritten block ahead of one that's never read.

**No coordination with vLLM.** This tool does not talk to vLLM at all; it only looks at the filesystem. A block could theoretically be deleted while vLLM is mid-read of it. The `--min-age` grace period mitigates this (don't touch anything accessed/written in the last N seconds) but is not a hard guarantee. Start with `--dry-run`, then a generous `--min-age`, in any environment where you can't tolerate an occasional cache-read race.

**Cross-instance sharing needs a consistent `PYTHONHASHSEED`.** This is a vLLM-side requirement (noted in vLLM's own KV offloading docs), not something this tool controls, but it affects whether your fleet's block filenames are actually comparable/shared in the first place.

**No object-store (S3) support yet.** vLLM's object-store secondary tier has the same unaddressed eviction gap, but S3 has no `atime` equivalent — a correct implementation there needs either S3 access logs/CloudTrail data events or an access-tracking proxy in front of it, which is a different and larger piece of work. This tool currently only covers the filesystem tier.

## Usage

```
pip install -e .
vllm-kv-evictor /path/to/fs-tier/root_dir --capacity 500GB --min-age 120 --dry-run
```

Once you've reviewed what it would evict:

```
vllm-kv-evictor /path/to/fs-tier/root_dir --capacity 500GB --min-age 120 --once
```

Run continuously (e.g. as a sidecar container or systemd service):

```
vllm-kv-evictor /path/to/fs-tier/root_dir --capacity 500GB --min-age 120 --poll-interval 30 --json-log
```

See `vllm-kv-evictor --help` for all options.

## Status

Early, narrowly-scoped, and built in direct response to the gap described in vllm-project/vllm#38260. Tested against synthetic block directories, not yet run against a live vLLM deployment — if you try it against a real one, issues and corrections are very welcome.

## License

Apache-2.0, matching vLLM's own license.
