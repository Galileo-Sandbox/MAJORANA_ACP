"""Inventory existing research artifacts without importing the training stack."""

import argparse
import ast
import hashlib
import json
import math
import struct
import subprocess
import zipfile
from pathlib import Path


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load_json(path):
    """Load historical JSON while preserving non-standard constants as strings."""
    return json.loads(path.read_text(), parse_constant=lambda value: value)


def inspect_array(archive, member):
    with archive.open(member) as stream:
        prefix = stream.read(8)
        if prefix[:6] != b"\x93NUMPY":
            raise ValueError(f"Invalid NPY member: {member}")
        length_format = "<H" if prefix[6] == 1 else "<I"
        length = struct.unpack(length_format, stream.read(struct.calcsize(length_format)))[0]
        header = ast.literal_eval(stream.read(length).decode("latin1"))
    result = {
        "shape": header["shape"],
        "dtype": header["descr"],
        "sha256_npy": hashlib.sha256(archive.read(member)).hexdigest(),
    }
    if header["shape"] == () and header["descr"] in ("<f8", "<i8"):
        raw = archive.read(member)
        result["value"] = struct.unpack("<d" if header["descr"] == "<f8" else "<q", raw[-8:])[0]
        if isinstance(result["value"], float) and not math.isfinite(result["value"]):
            if math.isnan(result["value"]):
                result["value"] = "NaN"
            elif result["value"] > 0:
                result["value"] = "Infinity"
            else:
                result["value"] = "-Infinity"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    if args.output.exists():
        parser.error("Output already exists; choose a new inventory filename.")
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and not set(p.relative_to(root).parts) & {".git", ".venv", "ml4phy-paper", "__pycache__"}
    )
    report = {
        "repository": str(root),
        "head_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "scope": "Local checkout only; no training, inference, or external artifact retrieval.",
        "configurations": [],
        "run_summaries": [],
        "prediction_caches": [],
        "audit_metrics": [],
        "source_files": [],
        "artifact_counts": {},
    }
    for suffix in (".ckpt", ".pt", ".h5", ".hdf5", ".npz"):
        report["artifact_counts"][suffix] = sum(p.suffix == suffix for p in paths)
    for path in paths:
        relative = str(path.relative_to(root))
        if path.suffix in (".yaml", ".yml"):
            report["configurations"].append({"path": relative, **fingerprint(path)})
        elif path.name == "run_summary.json":
            data = load_json(path)
            checkpoint = data.get("cnp_ckpt")
            report["run_summaries"].append(
                {
                    "path": relative,
                    **fingerprint(path),
                    "data": data,
                    "recorded_checkpoint_exists": bool(
                        checkpoint and (root / checkpoint).is_file()
                    ),
                }
            )
        elif path.suffix == ".npz":
            with zipfile.ZipFile(path) as archive:
                # Keep the audit utility compatible with the lab server's Python 3.8.
                arrays = {
                    m[:-4]: inspect_array(archive, m)
                    for m in archive.namelist()
                    if m.endswith(".npy")
                }
            report["prediction_caches"].append(
                {"path": relative, **fingerprint(path), "arrays": arrays}
            )
        elif path.name.endswith("_audit.json"):
            report["audit_metrics"].append(
                {"path": relative, **fingerprint(path), "data": load_json(path)}
            )
        elif path.suffix in (".py", ".md", ".toml", ".lock", ".ipynb"):
            report["source_files"].append({"path": relative, **fingerprint(path)})
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Inventoried {len(paths)} files; wrote {args.output}")


if __name__ == "__main__":
    main()
