import importlib.util
import json

import numpy as np


def load():
    spec = importlib.util.spec_from_file_location("v8", "finetune/timesfm3_v8_relay/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_formal(root, sizes=(10, 7, 9)):
    formal = root / "src" / "timesfm3_formal"
    shards = {"train": [], "val": []}
    for split in shards:
        (formal / split).mkdir(parents=True)
        for i, n in enumerate(sizes):
            name = f"shard_{i:05d}.npz"
            base = i * 100 + (0 if split == "train" else 50)
            contexts = np.arange(base, base + n, dtype=np.float32)[:, None, None] * np.ones((1, 120, 7), np.float32)
            np.savez_compressed(formal / split / name, contexts=contexts, targets=np.zeros((n, 10), np.float32))
            shards[split].append({"file": name, "samples": n})
    manifest = {"splits": {s: {"artifacts": {"shards": v}} for s, v in shards.items()}}
    (formal / "manifest.json").write_text(json.dumps(manifest))
    return formal


def ids(batch):
    return batch[0][:, 0, 0].tolist()


def test_stream_resume_is_exact(tmp_path):
    m = load()
    files = m.shard_files(make_formal(tmp_path), "train")
    full = m.ShardStream(files, 3, seed=1)
    reference = [ids(full.next_batch()) for _ in range(12)]  # 跨越多个分片与 epoch
    first = m.ShardStream(files, 3, seed=1)
    got = [ids(first.next_batch()) for _ in range(5)]
    resumed = m.ShardStream(files, 3, seed=1, cursor=json.loads(json.dumps(first.cursor())))
    got += [ids(resumed.next_batch()) for _ in range(7)]
    assert got == reference
    assert full.epoch >= 1


def test_epoch_covers_each_sample_once_minus_remainders(tmp_path):
    m = load()
    files = m.shard_files(make_formal(tmp_path), "train")
    stream = m.ShardStream(files, 3, seed=7)
    seen = []
    while stream.epoch == 0:
        batch = stream.next_batch()
        if stream.epoch == 0:
            seen += ids(batch)
    assert len(seen) == len(set(seen)) == 9 + 6 + 9  # 每片丢弃不足一个 batch 的余数


def test_val_subset_and_relay_discovery(tmp_path):
    m = load()
    formal = make_formal(tmp_path)
    c, t = m.load_val_subset(m.shard_files(formal, "val"), 5, seed=0)
    assert c.shape == (5, 120, 7) and t.shape == (5, 10)
    assert m.find_formal_root(tmp_path) == formal
    assert m.find_relay_state(tmp_path) is None
    for name, step in (("c01", 100), ("c02", 300)):
        relay = tmp_path / name / "relay"
        relay.mkdir(parents=True)
        (relay / "relay.json").write_text(json.dumps({"run_id": m.SWANLAB_RUN_ID, "global_step": step}))
    assert m.find_relay_state(tmp_path) == tmp_path / "c02" / "relay" / "state.pt"


def test_run_id_is_21_alnum():
    m = load()
    assert len(m.SWANLAB_RUN_ID) == 21 and m.SWANLAB_RUN_ID.isalnum()
