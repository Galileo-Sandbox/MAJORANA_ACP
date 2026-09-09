#!/usr/bin/env python3
"""Launch one paper evaluator job in the validated direct-server environment."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--phase", choices=("development", "final"), required=True)
    parser.add_argument("--context-seed", type=int, required=True)
    parser.add_argument("--dropout-seed", type=int)
    parser.add_argument("--n-context", type=int, default=2000)
    parser.add_argument("--n-mc", type=int, default=50)
    parser.add_argument("--max-context-per-pass", type=int, default=2000)
    parser.add_argument("--grid-spacing-kev", type=float, default=1.0)
    parser.add_argument("--run-kind", choices=("pilot", "evaluation"), default="evaluation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("run-id may contain only letters, numbers, hyphens, and underscores.")
    repo = args.repo.resolve()
    output_dir = repo / "ml4phy-paper/runs" / args.run_id
    if output_dir.exists():
        parser.error(f"Refusing to reuse output directory: {output_dir}")
    python = repo / ".venv/bin/python"
    evaluator = repo / "ml4phy-paper/scripts/evaluate_fixed_protocol.py"
    source_overlay = repo / "ml4phy-paper/local/resum-flex-edba6a"
    for path, description in (
        (python, "repository Python"),
        (evaluator, "paper evaluator"),
        (source_overlay / "core/__init__.py", "RESUM_FLEX source overlay"),
    ):
        if not path.exists():
            parser.error(f"Missing {description}: {path}")

    command = [
        str(python),
        str(evaluator),
        "--repo",
        str(repo),
        "--model-id",
        args.model_id,
        "--phase",
        args.phase,
        "--context-seed",
        str(args.context_seed),
        "--n-context",
        str(args.n_context),
        "--n-mc",
        str(args.n_mc),
        "--max-context-per-pass",
        str(args.max_context_per_pass),
        "--grid-spacing-kev",
        str(args.grid_spacing_kev),
        "--run-kind",
        args.run_kind,
        "--output-dir",
        str(output_dir),
    ]
    if args.dropout_seed is not None:
        command.extend(["--dropout-seed", str(args.dropout_seed)])
    env = os.environ.copy()
    prior_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(source_overlay)
        if not prior_pythonpath
        else f"{source_overlay}{os.pathsep}{prior_pythonpath}"
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = str(repo / "ml4phy-paper/local/matplotlib")
    if args.dry_run:
        print("PYTHONPATH=" + shlex.quote(env["PYTHONPATH"]))
        print(" ".join(shlex.quote(part) for part in command))
        return

    started = utc_now()
    result = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True, check=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stdout.log").write_text(result.stdout)
    (output_dir / "stderr.log").write_text(result.stderr)
    runner_record = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed" if result.returncode == 0 else "failed",
        "return_code": result.returncode,
        "start_time": started,
        "end_time": utc_now(),
        "command": command,
        "environment": {
            "pythonpath_source_overlay": "ml4phy-paper/local/resum-flex-edba6a",
            "python_dont_write_bytecode": True,
            "execution_mode": "direct local process",
        },
    }
    with (output_dir / "runner.json").open("x") as stream:
        json.dump(runner_record, stream, indent=2)
        stream.write("\n")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
