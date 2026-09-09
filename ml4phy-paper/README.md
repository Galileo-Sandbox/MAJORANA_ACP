# ML4PS 2026 Experiment Planning

This directory contains an evidence-first plan for a focused MAJORANA cut-acceptance paper. It does not launch training or alter the existing research workflow.

- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): Scope, verified findings, missing evidence, evaluation protocol, conditional experiment queue, and completion criteria.
- [SERVER_WORKFLOW.md](SERVER_WORKFLOW.md): Lab-server execution, Git handoff, artifact storage, and result-return requirements.
- [inventory.json](inventory.json): Machine-readable inventory of existing configurations, run summaries, prediction caches, array fingerprints, and source-file hashes.
- [audit_inventory.py](audit_inventory.py): Standard-library-only, read-only repository inspection. It writes an inventory only to an explicitly supplied, previously nonexistent output file.

The inventory was collected on September 8, 2026, from repository commit `c0664d1572812b4750fec0a117a19593c6b30be1`. The worktree was clean before this directory was added. File availability refers to this local checkout, not every machine or archive used by the authors.

To repeat the inventory without overwriting the original:

```sh
python3 ml4phy-paper/audit_inventory.py --repo . --output ml4phy-paper/inventory-next.json
```

All new paper-specific configurations, analysis adapters, figures, and experiment outputs should remain under `ml4phy-paper/`. Existing source, configurations, notebooks, results, and caches must remain unchanged. Training and expensive inference run on the lab server. The laptop is for planning, lightweight analysis, figure review, and manuscript compilation. Raw data and checkpoints stay on the server; selected lightweight results return through Git.

The canonical manuscript remains in `/Users/yuema137/Papers/MJD-Paper/`. This directory is an experiment workspace, not a second manuscript copy. All newly authored documentation, code, comments, and figure labels must be in English.
