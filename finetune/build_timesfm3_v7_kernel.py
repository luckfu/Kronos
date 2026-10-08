"""Package the self-contained TimesFM-3 runner and its budget plan."""

import argparse
import json
from pathlib import Path


def build(folder: Path) -> Path:
    source = Path(__file__).with_name("timesfm3_lora_finetune.py").read_text()
    plan = json.loads((folder / "run-plan.json").read_text())
    marker = "PACKAGED_RUN_PLAN = None"
    if source.count(marker) != 1:
        raise RuntimeError("Runner plan marker must occur exactly once")
    source = source.replace(marker, "PACKAGED_RUN_PLAN = " + repr(plan), 1)
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    target = folder / metadata["code_file"]
    compile(source, str(target), "exec")
    target.write_text(source)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    print(build(parser.parse_args().folder))
