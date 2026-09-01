"""Build the self-contained isolated Beta v3 bootstrap Kaggle kernel."""

import base64
import hashlib
import io
import json
import py_compile
import re
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "kaggle_v1_beta_minimal/Kronos"
V3_DATA = ROOT.parent / "data/a_share_full_market_v1_beta_symbol_holdout_90_10_v1"
RUNNER = ROOT / "kaggle_v3_bootstrap.py"
STAGING = ROOT / "kaggle_v3_bootstrap_kernel"
METADATA = {
    "id": "luckfu/kronos-beta-v3-dynamic-size-continuation-01",
    "title": "Kronos Beta v3 Dynamic Size Continuation 01",
    "code_file": RUNNER.name,
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": ["luckfu/kronos-a-share-full-market-v1-beta-120d"],
    "competition_sources": [],
    "kernel_sources": [
        "luckfu/kronos-beta-v3-base-dynamic-size-bootstrap",
    ],
}


def build_archive():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(SOURCE.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive.add(path, arcname=Path("Kronos") / path.relative_to(SOURCE))
        contract_files = {
            V3_DATA / "data_manifest.json": "data_manifest.json",
            V3_DATA / "symbol_split.csv": "symbol_split.csv",
        }
        for path, name in contract_files.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.add(path, arcname=Path("Kronos/finetune/v3_data_contract") / name)
        validation_root = (
            V3_DATA / "natural_validation_2025h2_2026h1_symbol_holdout_full_v1"
        )
        full_manifest = json.loads(
            (validation_root / "natural_validation_manifest.json").read_text()
        )
        all_records = [
            json.loads(line) for line in
            (validation_root / "natural_validation_samples.jsonl").read_text().splitlines()
            if line.strip()
        ]
        by_period = {}
        for record in all_records:
            by_period.setdefault(record["period"], []).append(record)
        if set(by_period) != {"2025H2", "2026H1"}:
            raise ValueError("Bootstrap validation must contain both half-year periods")
        records = []
        for period in ("2025H2", "2026H1"):
            pool = by_period[period]
            quota = 1000
            # Evenly sample each period while preserving the source ordering.
            indices = [round(i * (len(pool) - 1) / (quota - 1)) for i in range(quota)]
            records.extend(pool[index] for index in indices)
        records.sort(key=lambda item: (item["asof_date"], item["symbol"], item["start_index"]))
        records = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records]
        samples = ("\n".join(records) + "\n").encode()
        selection = full_manifest["selection"]
        selection["quick_samples"] = 2000
        selection["large_samples"] = 2000
        selection["samples_file_sha256"] = hashlib.sha256(samples).hexdigest()
        isolation = full_manifest["training_isolation"]
        isolation["training_candidate_overlap_samples"] = 0
        isolation["not_in_training_candidate_samples"] = 2000
        full_manifest["bootstrap_subset"] = {
            "samples": 2000,
            "period_samples": {"2025H2": 1000, "2026H1": 1000},
            "source_full_manifest_sha256": hashlib.sha256(
                (validation_root / "natural_validation_manifest.json").read_bytes()
            ).hexdigest(),
            "source_full_samples": 123982,
        }
        manifest = (
            json.dumps(full_manifest, indent=2, ensure_ascii=False) + "\n"
        ).encode()
        for name, content in (
            ("natural_validation_manifest.json", manifest),
            ("natural_validation_samples.jsonl", samples),
        ):
            info = tarfile.TarInfo(
                str(Path("Kronos/finetune/v3_data_contract") / name)
            )
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def main():
    encoded = base64.b64encode(build_archive()).decode("ascii")
    source = RUNNER.read_text()
    updated, count = re.subn(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n"""',
        f'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n{encoded}\n"""',
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("Unable to replace embedded Beta v3 source archive")
    RUNNER.write_text(updated)
    py_compile.compile(str(RUNNER), doraise=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNNER, STAGING / RUNNER.name)
    (STAGING / "kernel-metadata.json").write_text(
        json.dumps(METADATA, indent=2) + "\n"
    )
    print(json.dumps({"runner": str(RUNNER), "staging": str(STAGING)}, indent=2))


if __name__ == "__main__":
    main()
