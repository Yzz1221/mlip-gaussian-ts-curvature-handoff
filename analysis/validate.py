#!/usr/bin/env python3
"""Summarize paper-primary Opt+Freq and IRC Intended recovery for a dataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.audit_ts_freq_logs import classify as classify_opt_freq  # noqa: E402
from analysis.classify_gaussian_irc_intended import (  # noqa: E402
    ClassificationError,
    classify_intended,
)

DATASETS = {"gsm": "Gaussian_GSM_eqV2", "react_ot": "Gaussian_ReactOT_exact"}
WORKFLOWS = {
    "calcfc": "Gaussian_calcfc",
    "calcall": "Gaussian_calcall",
    "oneshot": "MLIP_OneShot_ReadFC",
    "external_calcall": "External_calcall",
}
REACTBENCH = ROOT / "third_party" / "ReactBench" / "ReactBench"
REFERENCE_ARCHIVE = ROOT / "data" / "transition1x_960.tar.gz"


def load_reactbench_connectivity_only() -> None:
    """Import the frozen adjacency source without unrelated ML calculator modules."""
    if "ReactBench" not in sys.modules:
        package = types.ModuleType("ReactBench")
        package.__path__ = [str(REACTBENCH)]
        sys.modules["ReactBench"] = package
    if "ReactBench.utils" not in sys.modules:
        package = types.ModuleType("ReactBench.utils")
        package.__path__ = [str(REACTBENCH / "utils")]
        sys.modules["ReactBench.utils"] = package


def expected_ids(dataset: str) -> list[str]:
    directory = ROOT / "data" / DATASETS[dataset] / "Gaussian_calcfc"
    ids = [child.name for child in directory.iterdir() if child.is_dir() and re.fullmatch(r"rxn\d+", child.name)]
    return sorted(ids, key=lambda value: int(value[3:]))


def read_reference(archive: tarfile.TarFile, reaction: str, directory: Path) -> Path:
    name = f"ts1x/{reaction}.xyz"
    member = archive.getmember(name)
    handle = archive.extractfile(member)
    if handle is None:
        raise RuntimeError(f"Cannot read {name} from {REFERENCE_ARCHIVE}")
    target = directory / f"{reaction}.xyz"
    target.write_bytes(handle.read())
    return target


def analyze(dataset: str, runs: Path, reactions: list[str]) -> dict[str, object]:
    if not REACTBENCH.is_dir():
        raise RuntimeError("ReactBench submodule missing: git submodule update --init --recursive")
    load_reactbench_connectivity_only()
    cases = []
    with tarfile.open(REFERENCE_ARCHIVE, "r:gz") as archive, tempfile.TemporaryDirectory() as temporary:
        reference_dir = Path(temporary)
        for reaction in reactions:
            case_dir = runs / reaction
            opt_log = case_dir / "opt_freq.log"
            irc_log = case_dir / "irc.log"
            opt_text = opt_log.read_text(encoding="utf-8", errors="replace") if opt_log.is_file() else ""
            opt = classify_opt_freq(opt_text)
            row: dict[str, object] = {
                "reaction": reaction,
                "opt_freq_success": bool(opt["paper_ok_minus_10"]),
                "opt_freq_reason": opt["paper_reason"],
                "imaginary_below_minus_10": opt["imaginary_below_minus_10"],
                "irc_log_present": irc_log.is_file(),
                "irc_category": None,
                "irc_intended": False,
            }
            if irc_log.is_file():
                try:
                    reference = read_reference(archive, reaction, reference_dir)
                    irc = classify_intended(irc_log, reference, REACTBENCH)
                    row["irc_category"] = irc["reaction_type"]
                    row["irc_intended"] = bool(irc["intended"])
                    row["irc_endpoint_points"] = irc["endpoint_point_numbers"]
                except (ClassificationError, KeyError, RuntimeError) as exc:
                    row["irc_category"] = "ClassificationError"
                    row["irc_reason"] = str(exc)
            cases.append(row)
    opt_count = sum(bool(row["opt_freq_success"]) for row in cases)
    intended_count = sum(bool(row["opt_freq_success"] and row["irc_intended"]) for row in cases)
    total = len(cases)
    return {
        "dataset": dataset,
        "denominator": total,
        "criterion": "paper_ok_minus_10",
        "opt_freq_success": opt_count,
        "opt_freq_success_rate": opt_count / total,
        "opt_freq_and_irc_intended": intended_count,
        "irc_intended_rate_of_all_inputs": intended_count / total,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--workflow", required=True, choices=WORKFLOWS)
    parser.add_argument("--runs", required=True, type=Path,
                        help="Directory containing rxn<ID>/opt_freq.log and optional irc.log")
    parser.add_argument("--reaction", help="Analyze one reaction instead of the full dataset")
    parser.add_argument("--output", type=Path, help="Write case-level JSON to this path")
    args = parser.parse_args()
    if not args.runs.is_dir():
        parser.error(f"Run directory does not exist: {args.runs}")
    reactions = expected_ids(args.dataset)
    if args.reaction:
        if args.reaction not in reactions:
            parser.error(f"{args.reaction} is not in {args.dataset}")
        reactions = [args.reaction]
    result = analyze(args.dataset, args.runs, reactions)
    result["workflow"] = args.workflow
    result["method_directory"] = WORKFLOWS[args.workflow]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
