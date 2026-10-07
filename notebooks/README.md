# Notebooks

`data_visualization.ipynb` is the historical interactive visualization
notebook. It reads compact arrays from `cache/inference/` for its portable
views, while some optional sections can load local predictions or checkpoints.

Review a cell before executing it: notebook source contains both visualization
and historical inference-loading paths. The ML4PS paper exports were generated
by scripts under `ml4phy-paper/` and should be used for the frozen paper
numbers. The notebook does not supersede those manifests or protocols.

For a no-training paper check, run
`ml4phy-paper/artifact/verify_artifact.py` instead of executing the entire
notebook.
