"""CPU-only DDP reproducer for nonpersistent buffer registration order."""
import json
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


class Probe(torch.nn.Module):
    def __init__(self, rank, canonical):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        values = {"full_attention_inv_freq": 160000.0, "sliding_attention_inv_freq": 10000.0}
        names = list(values) if rank == 0 else list(reversed(values))
        if canonical:
            names = sorted(names)
        for name in names:
            self.register_buffer(name, torch.tensor([values[name]]), persistent=False)

    def forward(self, x):
        return x * self.weight


def worker(rank, rendezvous, output, canonical):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    model = Probe(rank, canonical)
    before = {name: value.item() for name, value in model.named_buffers()}
    wrapped = DistributedDataParallel(model)
    wrapped(torch.ones(1))
    after = {name: value.item() for name, value in model.named_buffers()}
    Path(output, f"{rank}.json").write_text(json.dumps({
        "rank": rank, "before": before, "after": after,
        "state_dict_keys": list(model.state_dict()),
    }))
    dist.destroy_process_group()


def main():
    result = {"torch": torch.__version__, "scope": "CPU Gloo mechanism test; no financial model execution"}
    for canonical in (False, True):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(worker, args=(f"file://{directory}/rendezvous", directory, canonical),
                     nprocs=2, join=True)
            rows = [json.loads(Path(directory, f"{rank}.json").read_text()) for rank in range(2)]
            assert rows[0]["before"] == rows[1]["before"]
            assert rows[0]["state_dict_keys"] == rows[1]["state_dict_keys"] == ["weight"]
            assert (rows[0]["after"] == rows[1]["after"]) == canonical
            result["canonical_order" if canonical else "opposite_order"] = rows
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
