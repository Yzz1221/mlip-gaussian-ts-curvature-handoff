"""Command-line interfaces."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .audit import classify_opt_freq
from .gaussian_io import write_irc_input
from .workflow import prepare_handoff


def prepare_main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a one-shot ML Hessian ReadFC job")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", default="wB97X/6-31G(d)")
    parser.add_argument("--nproc", type=int, default=24)
    parser.add_argument("--memory")
    parser.add_argument("--max-cycles", type=int, default=150)
    parser.add_argument("--run-sp", action="store_true")
    args = parser.parse_args()
    result = prepare_handoff(
        args.input,
        args.output,
        method=args.method,
        nproc=args.nproc,
        memory=args.memory,
        max_cycles=args.max_cycles,
        run_sp=args.run_sp,
    )
    print(json.dumps(result, indent=2))
    return 0


def run_main() -> int:
    parser = argparse.ArgumentParser(description="Run one Gaussian input and write a sibling .log")
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    if not shutil.which("g16"):
        raise SystemExit("g16 not found in PATH")
    input_path = args.input.resolve()
    log_path = input_path.with_suffix(".log")
    with input_path.open("r", encoding="utf-8") as source, log_path.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            ["g16"], cwd=input_path.parent, stdin=source, stdout=output,
            stderr=subprocess.STDOUT, check=False
        )
    print(json.dumps({"input": str(input_path), "log": str(log_path), "exit_code": completed.returncode}))
    return completed.returncode


def audit_main() -> int:
    parser = argparse.ArgumentParser(description="Audit one Gaussian TS+Freq log")
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace") if args.log.is_file() else ""
    print(json.dumps(classify_opt_freq(text), indent=2))
    return 0


def write_irc_main() -> int:
    parser = argparse.ArgumentParser(description="Write a bidirectional Gaussian IRC input")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="wB97X/6-31G(d)")
    parser.add_argument("--nproc", type=int, default=24)
    parser.add_argument("--memory")
    parser.add_argument("--max-points", type=int, default=30)
    parser.add_argument("--step-size", type=int, default=15)
    args = parser.parse_args()
    write_irc_input(
        args.output,
        method=args.method,
        checkpoint=args.checkpoint,
        nproc=args.nproc,
        memory=args.memory,
        max_points=args.max_points,
        step_size=args.step_size,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(prepare_main())
