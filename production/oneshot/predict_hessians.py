#!/usr/bin/env python3
"""Predict a shard's Hessians with exactly one EquiformerV2 model load."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SOURCE_CODE = Path("/home/wuping/Experiment/TS+IRC/G_EF_Mlps_H/horm/mlip_readfc_scheme_experiments")
HORM_PROJECT = SOURCE_CODE.parent
sys.path.insert(0, str(SOURCE_CODE))
sys.path.insert(0, str(HORM_PROJECT))
from run_mlip_readfc_scheme import BOHR_TO_ANG, MlipHessianComputer, parse_ts_gjf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.manifest.open() as handle:
        entries = list(csv.DictReader(handle))
    if not entries:
        raise RuntimeError("empty manifest")

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
    os.environ["HORM_TORCH_NUM_THREADS"] = str(args.threads)
    rows = []
    total_start = time.perf_counter()
    load_start = time.perf_counter()
    with MlipHessianComputer() as computer:
        computer.horm.torch.set_num_threads(args.threads)
        model_load_s = time.perf_counter() - load_start
        for entry in entries:
            geom = parse_ts_gjf(Path(entry["source_gjf"]))
            start = time.perf_counter()
            coords_bohr = geom.coords_ang / BOHR_TO_ANG
            _, _, raw = computer.horm._forward_predict(
                computer.pm, computer.model_name, computer.device,
                geom.atomic_nums, coords_bohr, need_hessian=True,
            )
            inference_s = time.perf_counter() - start
            if raw is None:
                raise RuntimeError(f"{entry['dataset']}/{entry['reaction']}: model returned no Hessian")
            if hasattr(raw, "detach"):
                raw = raw.detach().cpu().numpy()
            hessian = np.asarray(raw, dtype=np.float64) * computer.horm.HESS_CONV
            hessian = 0.5 * (hessian + hessian.T)
            expected = (3 * len(geom.atomic_nums),) * 2
            if hessian.shape != expected or not np.all(np.isfinite(hessian)):
                raise RuntimeError(f"{entry['dataset']}/{entry['reaction']}: invalid Hessian {hessian.shape}")
            dest = args.output / entry["dataset"]
            dest.mkdir(exist_ok=True)
            artifact = dest / f"{entry['reaction']}.npz"
            np.savez(
                artifact,
                reaction_id=np.asarray(entry["reaction"]),
                dataset=np.asarray(entry["dataset"]),
                atomic_numbers=geom.atomic_nums,
                coordinates_angstrom=geom.coords_ang,
                hessian_hartree_bohr2=hessian,
            )
            rows.append({
                "dataset": entry["dataset"], "reaction": entry["reaction"],
                "natoms": len(geom.atomic_nums), "threads": args.threads,
                "device": computer.device, "inference_time_s": f"{inference_s:.9f}",
                "artifact": str(artifact.relative_to(args.output)),
            })
            print(f"{entry['dataset']}/{entry['reaction']} threads={args.threads} inference_s={inference_s:.6f}", flush=True)

    with (args.output / "timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "threads": args.threads,
        "reaction_count": len(rows),
        "device": rows[0]["device"],
        "model_load_time_s": model_load_s,
        "model_load_amortized_per_reaction_s": model_load_s / len(rows),
        "predictor_total_time_s": time.perf_counter() - total_start,
        "checkpoint": os.environ["HORM_CHECKPOINT"],
        "checkpoint_sha256": sha256(Path(os.environ["HORM_CHECKPOINT"])),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
