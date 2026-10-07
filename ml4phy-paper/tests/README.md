# Paper export tests

These tests exercise the presentation and regional-coverage exporters without
starting training or inference.

- `test_paper_presentation_export.py` covers Wilson intervals, supported-bin
  handling, regional diagnostics, and missing uncertainty.
- `test_paper_coverage_export.py` covers regional membership, boundary bins,
  support exclusions, threshold monotonicity, and paired support.

Run them from the repository root with:

```bash
PYTHONPATH=ml4phy-paper/scripts:. uv run pytest ml4phy-paper/tests -q
```

Mechanism-control tests are documented under
`../review_followup_20260910/tests/` and the artifact verifier has its own tests
under `../artifact/tests/`.
