"""Evaluation pipeline: load a checkpoint, score the test set, save metrics."""

from majorana_acp.eval.evaluator import (
    compute_metrics,
    evaluate,
    load_checkpoint,
    run_inference,
    save_predictions,
)

__all__ = [
    "compute_metrics",
    "evaluate",
    "load_checkpoint",
    "run_inference",
    "save_predictions",
]
