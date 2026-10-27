import os
import time
from datetime import datetime, timedelta, timezone

import pytest

import evictor

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")
from moto import mock_aws


def _make_block(rank_dir, name, content=b"x", age_seconds=0):
    p = rank_dir / name
    p.write_bytes(content)
    t = time.time() - age_seconds
    os.utime(p, (t, t))
    return p


def test_scan_blocks_finds_files_under_rank_dirs_only(tmp_path):
    r0 = tmp_path / "_r0"
    r0.mkdir()
    _make_block(r0, "a.blk", b"aaaa")
    other = tmp_path / "not_a_rank_dir"
    other.mkdir()
    (other / "ignored.blk").write_bytes(b"zz")

    blocks = evictor.scan_blocks(tmp_path)
    assert len(blocks) == 1
    assert blocks[0].path.name == "a.blk"
    assert blocks[0].size == 4


def test_scan_blocks_recurses_into_subdirectories(tmp_path):
    r0 = tmp_path / "_r0"
    (r0 / "nested").mkdir(parents=True)
    _make_block(r0 / "nested", "b.blk", b"yy")
    blocks = evictor.scan_blocks(tmp_path)
    assert len(blocks) == 1


def test_plan_eviction_under_capacity_evicts_nothing():
    blocks = [evictor.Block(path="a", size=10, atime=0, mtime=0)]
    assert evictor.plan_eviction(blocks, capacity_bytes=100, low_water_bytes=90,
                                  min_age_seconds=0, now=1000) == []


def test_plan_eviction_stops_at_low_water_mark():
    now = 10_000
    old = evictor.Block(path="old", size=40, atime=1, mtime=1)
    mid = evictor.Block(path="mid", size=40, atime=500, mtime=500)
    new = evictor.Block(path="new", size=40, atime=9000, mtime=9000)
    plan = evictor.plan_eviction([old, mid, new], capacity_bytes=100, low_water_bytes=50,
                                  min_age_seconds=0, now=now)
    # total=120 > capacity=100 -> evict oldest-first until total <= low_water=50
    # evict "old" (120->80, still >50) then "mid" (80->40, <=50) -> stop. "new" survives.
    assert [b.path for b in plan] == ["old", "mid"]


def test_plan_eviction_respects_min_age_grace_period():
    now = 10_000
    recent = evictor.Block(path="recent", size=100, atime=now - 5, mtime=now - 5)
    plan = evictor.plan_eviction([recent], capacity_bytes=10, low_water_bytes=5,
                                  min_age_seconds=60, now=now)
    assert plan == []  # over capacity, but too young to touch


def test_plan_eviction_uses_mtime_when_recency_is_mtime():
    now = 10_000
    # atime says "new" is oldest, mtime says "new" is newest -- recency="mtime" must follow mtime
    a = evictor.Block(path="a", size=60, atime=9000, mtime=1)
    b = evictor.Block(path="b", size=60, atime=1, mtime=9000)
    plan = evictor.plan_eviction([a, b], capacity_bytes=100, low_water_bytes=65,
                                  min_age_seconds=0, now=now, recency="mtime")
    assert [x.path for x in plan] == ["a"]  # a has the oldest mtime; evicting it (120->60) clears low_water=65


def test_evict_dry_run_reports_without_deleting(tmp_path):
    p = tmp_path / "f.blk"
    p.write_bytes(b"1234")
    block = evictor.Block(path=p, size=4, atime=0, mtime=0)
    freed, errors = evictor.evict([block], dry_run=True)
    assert freed == 4
    assert errors == []
    assert p.exists()


def test_evict_real_deletes_and_frees_bytes(tmp_path):
    p = tmp_path / "f.blk"
    p.write_bytes(b"1234")
    block = evictor.Block(path=p, size=4, atime=0, mtime=0)
    freed, errors = evictor.evict([block], dry_run=False)
    assert freed == 4
    assert errors == []
    assert not p.exists()


def test_evict_missing_file_reports_error_not_crash(tmp_path):
    p = tmp_path / "missing.blk"
    block = evictor.Block(path=p, size=4, atime=0, mtime=0)
    freed, errors = evictor.evict([block], dry_run=False)
    assert freed == 0
    assert len(errors) == 1


def test_run_once_end_to_end_dry_run(tmp_path):
    r0 = tmp_path / "_r0"
    r0.mkdir()
    _make_block(r0, "old.blk", b"x" * 50, age_seconds=1000)
    _make_block(r0, "new.blk", b"x" * 50, age_seconds=0)
    result = evictor.run_once(tmp_path, capacity_bytes=60, low_water_bytes=50,
                               min_age_seconds=0, recency="atime", dry_run=True)
    assert result["scanned_files"] == 2
    assert result["total_bytes_before"] == 100
    assert result["evicted_files"] == 1
    assert result["bytes_freed"] == 50
    assert (r0 / "old.blk").exists()  # dry run: nothing actually deleted
    assert (r0 / "new.blk").exists()


def test_run_once_end_to_end_real_eviction(tmp_path):
    r0 = tmp_path / "_r0"
    r0.mkdir()
    _make_block(r0, "old.blk", b"x" * 50, age_seconds=1000)
    _make_block(r0, "new.blk", b"x" * 50, age_seconds=0)
    result = evictor.run_once(tmp_path, capacity_bytes=60, low_water_bytes=50,
                               min_age_seconds=0, recency="atime", dry_run=False)
    assert result["evicted_files"] == 1
    assert not (r0 / "old.blk").exists()
    assert (r0 / "new.blk").exists()


def test_parse_size_variants():
    assert evictor.parse_size("500") == 500
    assert evictor.parse_size("1KB") == 1000
    assert evictor.parse_size("1KiB") == 1024
    assert evictor.parse_size("2.5GB") == int(2.5 * 1000**3)
    assert evictor.parse_size("1TiB") == 1024**4


def test_parse_size_rejects_garbage():
    with pytest.raises(Exception):
        evictor.parse_size("not-a-size")


def test_detect_atime_support_on_a_normal_tmp_mount(tmp_path):
    # tmp_path is almost always on a normal (non-noatime) filesystem in CI/dev
    # containers; this is a real, not mocked, filesystem check.
    result = evictor.detect_atime_support(tmp_path)
    assert isinstance(result, bool)


# ─────────────────────────────────────────────
# S3Backend — vLLM's object-store secondary tier (moto-mocked, no real AWS)
# ─────────────────────────────────────────────
@pytest.fixture
def s3_bucket():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="kvcache-test")
        yield client, "kvcache-test"


def _put_s3_object(client, bucket, key, body=b"x", age_days=0):
    client.put_object(Bucket=bucket, Key=key, Body=body)
    if age_days:
        # moto's in-memory backend lets us directly backdate LastModified by
        # reaching into the mocked object store rather than sleeping real time.
        from moto.s3.models import s3_backends
        backend = s3_backends["123456789012"]["aws"]
        key_obj = backend.buckets[bucket].keys.get(key)
        if key_obj is not None:
            key_obj.last_modified = datetime.now(timezone.utc) - timedelta(days=age_days)


def test_s3_backend_scan_lists_objects_with_last_modified_as_recency(s3_bucket):
    client, bucket = s3_bucket
    _put_s3_object(client, bucket, "blocks/a", b"aaaa")
    backend = evictor.S3Backend(bucket)
    blocks = backend.scan()
    assert len(blocks) == 1
    assert blocks[0].path == "blocks/a"
    assert blocks[0].size == 4
    assert blocks[0].atime == blocks[0].mtime  # S3 only has one timestamp -- write time


def test_s3_backend_respects_prefix_filter(s3_bucket):
    client, bucket = s3_bucket
    _put_s3_object(client, bucket, "kv/a", b"x")
    _put_s3_object(client, bucket, "other/b", b"yy")
    backend = evictor.S3Backend(bucket, prefix="kv/")
    blocks = backend.scan()
    assert [b.path for b in blocks] == ["kv/a"]


def test_s3_backend_detect_recency_mode_is_always_mtime(s3_bucket):
    client, bucket = s3_bucket
    backend = evictor.S3Backend(bucket)
    assert backend.detect_recency_mode() == "mtime"  # no probe can change this -- S3 has no atime


def test_s3_backend_delete_dry_run_does_not_delete(s3_bucket):
    client, bucket = s3_bucket
    _put_s3_object(client, bucket, "a", b"1234")
    backend = evictor.S3Backend(bucket)
    blocks = backend.scan()
    freed, errors = backend.delete(blocks, dry_run=True)
    assert freed == 4
    assert errors == []
    assert len(backend.scan()) == 1  # still there


def test_s3_backend_delete_removes_objects_and_sums_freed_bytes(s3_bucket):
    client, bucket = s3_bucket
    _put_s3_object(client, bucket, "a", b"1234")
    _put_s3_object(client, bucket, "b", b"56")
    backend = evictor.S3Backend(bucket)
    blocks = backend.scan()
    freed, errors = backend.delete(blocks, dry_run=False)
    assert freed == 6
    assert errors == []
    assert backend.scan() == []


def test_s3_backend_delete_chunks_over_1000_keys(s3_bucket):
    client, bucket = s3_bucket
    n = 1500
    for i in range(n):
        _put_s3_object(client, bucket, f"blk/{i}", b"x")
    backend = evictor.S3Backend(bucket, prefix="blk/")
    blocks = backend.scan()
    assert len(blocks) == n
    freed, errors = backend.delete(blocks, dry_run=False)
    assert freed == n
    assert errors == []
    assert backend.scan() == []


def test_run_once_backend_end_to_end_with_s3(s3_bucket):
    client, bucket = s3_bucket
    _put_s3_object(client, bucket, "old", b"x" * 50, age_days=10)
    _put_s3_object(client, bucket, "new", b"x" * 50, age_days=0)
    backend = evictor.S3Backend(bucket)
    result = evictor.run_once_backend(backend, capacity_bytes=60, low_water_bytes=50,
                                       min_age_seconds=0, recency="mtime", dry_run=False)
    assert result["evicted_files"] == 1
    remaining = {b.path for b in backend.scan()}
    assert remaining == {"new"}  # "old" (write time 10 days ago) evicted, "new" kept
