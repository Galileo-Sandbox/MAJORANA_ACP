# Repository tests

The root test suite covers the reusable `majorana_acp` package.

| Path | Coverage |
| --- | --- |
| `analysis/` | Numerical metrics and regional masks. |
| `cut_acceptance/` | Efficiency-model configuration, samplers, positional encoding, model parity, and pipeline behavior. |
| `test_data.py` | Dataset preprocessing and loading. |
| `test_models.py` | Waveform-model registry and forward behavior. |
| `test_trainer.py`, `test_loss.py` | Optimization and loss behavior. |
| `test_evaluator.py` | Evaluation and prediction export. |

Run all package tests with:

```bash
PYTHONPATH=ml4phy-paper/local/resum-flex-edba6a:. uv run pytest
```

Paper-export and mechanism-control tests live beside their code under
`ml4phy-paper/tests/` and `ml4phy-paper/review_followup_20260910/tests/`.
