#!/usr/bin/env python3
"""Validate both source populations and build seven balanced, paired shards."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path("/home/wuping/Experiment/TS+IRC")
HERE = Path(__file__).resolve().parents[1]
NSHARDS = 7
DATASETS = {
    "gsm": ("Gaussian_GSM_eqV2", 871),
    "reactot": ("Gaussian_ReactOT_exact", 960),
}
SOURCE_WORKFLOWS = ("Gaussian_calcfc", "Gaussian_calcall", "mlip_oneshot_Readfc")


def rxn_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 10**18, name)


def reaction_ids(path: Path) -> set[str]:
    return {
        child.name for child in path.iterdir()
        if child.is_dir() and child.name.startswith("rxn")
    }


def atom_count(gjf: Path) -> int:
    lines = gjf.read_text(errors="replace").splitlines()
    charge_idx = next(
        i for i, line in enumerate(lines)
        if re.fullmatch(r"\s*[+-]?\d+\s+\d+\s*", line)
    )
    count = 0
    for line in lines[charge_idx + 1:]:
        if not line.strip() or len(line.split()) < 4:
            break
        count += 1
    if count == 0:
        raise RuntimeError(f"no atoms parsed from {gjf}")
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    manifest_dir = HERE / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    audit: dict[str, object] = {"nshards": NSHARDS, "datasets": {}}

    for short, (dirname, expected) in DATASETS.items():
        base = ROOT / dirname
        sets = {wf: reaction_ids(base / wf) for wf in SOURCE_WORKFLOWS}
        reference = sets["Gaussian_calcfc"]
        if len(reference) != expected:
            raise RuntimeError(f"{short}: expected {expected} CalcFC IDs, found {len(reference)}")
        for workflow, ids in sets.items():
            if ids != reference:
                missing = sorted(reference - ids, key=rxn_key)
                extra = sorted(ids - reference, key=rxn_key)
                raise RuntimeError(
                    f"{short}/{workflow}: ID mismatch; missing={missing[:10]} extra={extra[:10]}"
                )
        for rid in sorted(reference, key=rxn_key):
            gjf = base / "Gaussian_calcfc" / rid / "TS+Freq" / "ts+freq.gjf"
            xyz = base / "Gaussian_calcfc" / rid / "ts_guess.xyz"
            irc = base / "Gaussian_calcfc" / rid / "IRC" / "irc.gjf"
            for required in (gjf, xyz, irc):
                if not required.is_file():
                    raise RuntimeError(f"missing required source file: {required}")
            natoms = atom_count(gjf)
            rows.append({
                "dataset": short,
                "dataset_dir": dirname,
                "reaction": rid,
                "natoms": natoms,
                "weight": natoms**3,
                "source_gjf": str(gjf),
            })
        audit["datasets"][short] = {
            "directory": dirname,
            "expected": expected,
            "validated": len(reference),
            "source_workflows": list(SOURCE_WORKFLOWS),
        }

    # Longest-processing-time greedy packing by N_atoms^3. Each reaction stays
    # on one node for every workflow and core count.
    shards: list[list[dict[str, object]]] = [[] for _ in range(NSHARDS)]
    loads = [0] * NSHARDS
    for row in sorted(rows, key=lambda item: (-int(item["weight"]), str(item["dataset"]), rxn_key(str(item["reaction"])))):
        shard = min(range(NSHARDS), key=lambda idx: (loads[idx], len(shards[idx]), idx))
        row = dict(row)
        row["shard"] = shard
        shards[shard].append(row)
        loads[shard] += int(row["weight"])

    fields = ["dataset", "dataset_dir", "reaction", "natoms", "weight", "source_gjf", "shard"]
    all_path = manifest_dir / "all_reactions.csv"
    with all_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted((row for shard in shards for row in shard), key=lambda x: (str(x["dataset"]), rxn_key(str(x["reaction"])))))

    shard_meta = []
    for idx, shard_rows in enumerate(shards):
        path = manifest_dir / f"shard_{idx}.csv"
        ordered = sorted(shard_rows, key=lambda x: (str(x["dataset"]), rxn_key(str(x["reaction"]))))
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(ordered)
        counts = {name: sum(row["dataset"] == name for row in ordered) for name in DATASETS}
        shard_meta.append({
            "shard": idx,
            "count": len(ordered),
            "counts_by_dataset": counts,
            "weight": loads[idx],
            "manifest": path.name,
            "sha256": sha256(path),
        })

    flattened = {(str(row["dataset"]), str(row["reaction"])) for shard in shards for row in shard}
    expected_pairs = {(str(row["dataset"]), str(row["reaction"])) for row in rows}
    if flattened != expected_pairs or sum(len(shard) for shard in shards) != len(rows):
        raise RuntimeError("shard union/uniqueness validation failed")
    audit.update({
        "total_reactions": len(rows),
        "shards": shard_meta,
        "all_manifest": all_path.name,
        "all_manifest_sha256": sha256(all_path),
        "balancing_weight": "natoms^3",
    })
    (manifest_dir / "manifest_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
