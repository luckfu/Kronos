"""Read-only synthetic audit of the checked-in Kairos relay cursor code."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assigns(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def run_r2_cursor(
    path: Path, sizes: list[int], limit: int, start: int = 0, group_start: int = 0
) -> dict:
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    outer = copy.deepcopy(next(n for n in main.body if isinstance(n, ast.While)))
    inner = next(n for n in outer.body if isinstance(n, ast.While))
    boundary = next(i for i, n in enumerate(inner.body) if assigns(n, "global_segment_index"))
    reset = next(i for i, n in enumerate(inner.body) if assigns(n, "segment_rows"))
    # Keep both original loops and their exact cursor updates. Replace only model,
    # validation and checkpoint I/O with recording of synthetic row identities.
    record = ast.parse(
        "seen.extend(segment_rows)\n"
        "segments.append(list(segment_rows))\n"
        "processed += len(segment_rows)\n"
    ).body
    cursor_tail = inner.body[reset:reset + 4]
    assert assigns(cursor_tail[0], "segment_rows")
    assert isinstance(cursor_tail[3], ast.If)
    inner.body = inner.body[:boundary] + record + cursor_tail

    groups = [[f"{g}:{i}" for i in range(size)] for g, size in enumerate(sizes)]

    class Rows:
        def __init__(self, values):
            self.values = values

        def to_pylist(self):
            return self.values

    class Parquet:
        def read_row_group(self, group_id, columns):
            return Rows(groups[group_id])

    scope = {
        "group_order": list(range(len(groups))),
        "group_order_pos": group_start,
        "row_offset": start,
        "segments_this_run": 0,
        "completed_segments": 0,
        "run_segment_limit": limit,
        "SEGMENT_SAMPLES": 4,
        "segment_rows": [],
        "columns": [],
        "train_pq": Parquet(),
        "shuffle_group_rows": lambda rows, _: rows,
        "seen": [],
        "segments": [],
        "processed": 0,
    }
    code = ast.fix_missing_locations(ast.Module(body=[outer], type_ignores=[]))
    exec(compile(code, str(path), "exec"), scope)
    expected = [row for group in groups[group_start:] for row in group][start:][:limit * 4]
    seen = scope["seen"]
    return {
        "source": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "group_sizes": sizes,
        "initial_offset": start,
        "initial_group": group_start,
        "expected": expected,
        "observed": seen,
        "duplicate_visits": len(seen) - len(set(seen)),
        "missing_expected_rows": sorted(set(expected) - set(seen)),
        "cursor": [scope["group_order_pos"], scope["row_offset"]],
        "matches_contiguous_expected": seen == expected,
    }


def run_r1_partition() -> list[dict]:
    results = []
    for groups in (1, 7, 8, 9, 177, 181):
        for chunk in range(8):
            path = ROOT / (
                f"finetune/kaggle_modernbert_decision_full_chunk{chunk + 1}/"
                f"kaggle_modernbert_decision_full_chunk{chunk + 1}.py"
            )
            tree = ast.parse(path.read_text())
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            selection = [n for n in main.body if any(
                assigns(n, name) for name in ("chunk_start", "chunk_end", "chunk_groups")
            )]
            scope = {
                "train_pq": type("Parquet", (), {"num_row_groups": groups})(),
                "CHUNK_INDEX": chunk, "CHUNK_COUNT": 8,
                "group_order": list(reversed(range(groups))),
            }
            exec(compile(ast.Module(body=selection, type_ignores=[]), str(path), "exec"), scope)
            if chunk == 0:
                joined = []
            joined.extend(scope["chunk_groups"])
            checkpoint_if = next(
                n for n in main.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == "checkpoint.exists()"
            )
            resume_branch = next(
                n for n in checkpoint_if.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == "saved_chunk == CHUNK_INDEX"
            )
            branch = compile(ast.Module(body=[resume_branch], type_ignores=[]), str(path), "exec")
            for saved_chunk, saved_pos, expected in ((chunk - 1, 999, 0), (chunk, 2, 2)):
                state = {
                    "CHUNK_INDEX": chunk, "saved_chunk": saved_chunk,
                    "saved": {"chunk_pos": saved_pos}, "start_pos": 0,
                }
                exec(branch, state)
                assert state["start_pos"] == expected
        assert joined == list(reversed(range(groups)))
        results.append({"groups": groups, "no_partition_overlap_or_gap": True})
    return results


def main() -> None:
    cases = []
    for chunk in range(1, 5):
        path = ROOT / f"finetune/kaggle_kairos_r2_chunk{chunk}/kaggle_kairos_r2_chunk{chunk}.py"
        cases.append(run_r2_cursor(path, [5, 5, 3], 3))
        cases.append(run_r2_cursor(path, [5, 5, 3], 2, start=4))
    assert all(case["matches_contiguous_expected"] for case in cases)
    print(json.dumps({
        "purpose": "synthetic control-flow audit only; no model execution",
        "r1_current_partition_and_resume_cases": run_r1_partition(),
        "r2_fixed_loop_coverage_cases": cases,
        "limits": [
            "Does not verify historical Kaggle versions or checkpoint tensor states.",
            "R2 synthetic group sizes preserve the non-divisible group/segment boundary.",
            "A successful exit verifies synthetic cursor coverage, not optimizer equivalence.",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
