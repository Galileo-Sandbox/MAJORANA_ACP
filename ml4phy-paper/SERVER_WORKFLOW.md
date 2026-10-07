# Lab Server Execution and Git Handoff

## Operating model

The laptop is used to inspect existing evidence, design experiments, implement and review paper-specific analysis code, examine returned results, and compile the manuscript. All model training and expensive inference run on the lab server. Do not install the CUDA training stack or recreate missing training artifacts on the laptop.

The MAJORANA_ACP repository carries the experiment plan, reviewed scripts/configurations, and selected lightweight results. The manuscript remains a separate local TeX project. All new experimental work belongs under `ml4phy-paper/`; existing research files remain unchanged.

The sequence is:

```text
Laptop: plan + reviewed execution code -> commit + push
Server: clone/pull -> audit existing artifacts -> resolve gaps
Server: run only necessary experiments -> export lightweight results
Server: commit + push results
Laptop: pull -> inspect evidence -> update manuscript
```

A server push is required before a laptop pull can retrieve results. Git does not transfer uncommitted outputs or files excluded by ignore rules.

## Current readiness

Available now: detailed experiment plan, historical artifact inventory, and a standard-library inventory utility. Not yet available: the paper-specific evaluator with a frozen threshold and explicit event identities, matched experiment configuration bundle, batch runner, and result exporter.

Cloning the repository does not by itself make the proposed experiments runnable. Implement and review those missing pieces under `ml4phy-paper/` after the server artifact audit establishes what is reusable. Do not substitute the legacy notebook's default execution for the proposed evaluation protocol.

## 1. Publish the planning package

The existing remote is `git@github.com:Galileo-Sandbox/MAJORANA_ACP.git`. Use a dedicated branch, proposed name `ml4phy-paper`, after checking whether it already exists. Review and commit the planning files before cloning that branch on the server. No commit or push was performed while writing this workflow.

Example first publication, from a clean or reviewed checkout with no existing branch of that name:

```sh
git switch -c ml4phy-paper
git add ml4phy-paper/
git diff --cached --stat
git diff --cached
git commit -m "Add ML4PS experiment plan and server workflow"
git push -u origin ml4phy-paper
```

Review staged files before committing. If the branch exists, switch to it rather than creating it again. Never force-push to resolve divergence.

## 2. Clone or update on the lab server

For a fresh checkout after publication:

```sh
git clone --branch ml4phy-paper git@github.com:Galileo-Sandbox/MAJORANA_ACP.git
cd MAJORANA_ACP
git rev-parse HEAD
git status --short
```

If the server already has a checkout containing historical runs, preserve it and its untracked artifacts. Inspect its branch and changes before updating. Use an additional checkout if necessary; do not replace the existing directory or delete old results. This is especially important because the laptop checkout contains no weights or HDF5 files, while the original server may still contain them.

## 3. Audit before environment changes or training

Read `EXPERIMENT_PLAN.md`, especially Sections 3-5. Locate the historical `runs/`, `results/`, v9/Cell 17 checkpoints, full-test classifier predictions, and the exact RESUM_FLEX installation. Compare checkpoint metadata with YAML values; do not assume the current YAML is the one used to train a recovered model.

On the server, first write a new inventory into the ignored local area. Choose a new filename if it already exists:

```sh
mkdir -p ml4phy-paper/local
python3 ml4phy-paper/audit_inventory.py --repo . --output ml4phy-paper/local/server-inventory-001.json
```

This utility inventories the checkout, not external data storage. It excludes `ml4phy-paper/` so it does not recursively catalog its own outputs. Record relevant external or recovered artifacts separately, with their hashes and locations in a local manifest. The utility hashes NPZ members and may take longer on a server with many large archives; it does not train models or modify existing files.

Before returning an inventory through Git, remove unnecessary machine-specific paths and check that it contains no restricted event-level data. Keep the full machine-local manifest in `local/` and return a portable summary in `reports/artifact_recovery.md`.

Confirm the existing Python, PyTorch/CUDA/driver combination and upstream `core`/`schemas` package revision. Prefer the known working lab environment when reproducible. Do not blindly replace it with a new environment or assume `uv sync` alone installs RESUM_FLEX. Record the resolved environment before changing anything.

## 4. Turn the gap decision into runnable experiments

Complete the cache analysis and artifact recovery first. Write `reports/gap_decision.md` before new training. Each proposed job must identify the missing evidence it supplies, why an existing run cannot supply it, and whether inference alone is sufficient.

Prepare and review the following under this directory:

- Portable experiment configurations with logical artifact identifiers; machine-specific paths resolve from `local/`, not hard-coded laptop paths.
- An evaluator accepting a fixed calibration threshold, explicit context/target IDs, independent randomness controls, and a unique output directory.
- Matched configurations only for genuinely missing controls.
- A runner that fails on missing inputs or existing output files, records the exact source commit, and supports the lab's actual scheduler or job-launch mechanism.
- A compact result exporter that validates schema, counts, provenance, and file sizes.

Do not invent scheduler directives, GPU counts, or runtime estimates before inspecting the lab environment. Start with a small inference pilot; perform a training pilot only if required. Preserve the planned conditional experiment queue rather than launching every possible variant.

## 5. What stays on the server and what returns through Git

| Artifact | Location | Git policy |
|---|---|---|
| Raw waveforms and classifier HDF5 predictions | Existing storage, or `data/` / `artifacts/` | Keep server-side |
| Checkpoints, optimizer state, verbose logs, temporary caches | `runs/`, `artifacts/`, `logs/`, `tmp/` | Ignored; keep server-side |
| Machine-specific paths and full local manifests | `local/` | Ignored |
| Reviewed portable configurations and scripts | `configs/`, `scripts/` | Commit |
| Source/dependency revisions, hashes, seeds, protocol | `manifests/` | Commit portable provenance |
| Aggregate metrics and paired-run summaries | `tables/`, `reports/` | Commit |
| Selected publication figures | `figures/` | Commit compact PDF/PNG/SVG outputs |
| Compact prediction curves required for local plotting | `exports/` | Commit only after size/content review |

All table paths are relative to `ml4phy-paper/`. Its own `.gitignore` protects new heavy-output directories without changing the repository's original ignore file. Avoid `git add -f` for model or data artifacts. Review exports explicitly; Git is not the transport for full training datasets or checkpoints.

Every completed run should produce a lightweight summary containing:

- Run ID, status (`completed`, `failed`, or `partial`), source commit, and start/end times.
- Resolved portable configuration and exact command; dependency/environment identifiers.
- Classifier, CNP checkpoint, training-density pool, and split-manifest hashes.
- Threshold value and calibration provenance; population and energy range.
- Train/context/target counts, per-pass context cap, training/context/dropout seeds, and MC-pass count.
- Metric definitions, regional support counts, exclusions, and numeric results.
- Runtime, device, and peak memory where measured.
- Output hashes and the logical location of server-only artifacts needed for reproduction.

Return aggregate results for every planned run, including failures and unfavorable results. Compact curve exports must include energies, predictions, model/version identity, and protocol metadata sufficient to reproduce figures without GPU access. Do not merge incompatible cached curves and newly computed target statistics.

## 6. Publish results and retrieve them locally

Commit outputs only after the job has finished writing them. Review selected files explicitly; for example, stage the actual completed report and table paths, inspect `git diff --cached --stat` and `git diff --cached`, then commit and push the branch. Do not stage continuously changing run directories.

On the laptop, inspect the worktree and then update the shared branch:

```sh
git status --short
git pull --ff-only
```

If the branches have diverged, inspect and reconcile the actual changes. Do not reset away local work or force-push. Prefer alternating handoffs: publish local planning/code changes before the server phase, then retrieve server result commits before the next local edit phase.

After pulling, validate run completeness, provenance, metric definitions, and agreement between figures and tables before incorporating numbers into the manuscript. A finished job is not automatically a validated scientific result.
