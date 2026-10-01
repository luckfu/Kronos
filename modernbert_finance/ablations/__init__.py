"""Cheap Kairos / ModernBERT ablations (CPU-first)."""

from modernbert_finance.ablations.alignment import verify_feature_label_alignment
from modernbert_finance.ablations.label_shuffle import run_label_shuffle_sanity
from modernbert_finance.ablations.simple_baseline import run_simple_baseline

__all__ = [
    "verify_feature_label_alignment",
    "run_label_shuffle_sanity",
    "run_simple_baseline",
]
