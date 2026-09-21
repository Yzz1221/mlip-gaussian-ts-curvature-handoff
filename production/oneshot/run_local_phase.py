#!/usr/bin/env python3
"""Run one full-node Gaussian phase using node-local working files."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path("/home/wuping/Experiment/TS+IRC")
SOURCE_CODE = ROOT / "G_EF_Mlps_H/horm/mlip_readfc_scheme_experiments"
sys.path.insert(0, str(SOURCE_CODE))
from run_mlip_readfc_scheme import (
    BOHR_TO_ANG, coord_lines, parse_ts_gjf, read_fchk_array,
    read_fchk_int_array, replace_or_insert_force_constants,
)

METHOD = "wB97X/6-31G(d)"
ELAPSED_RE = re.compile(r"Elapsed time:\s+(\d+) days\s+(\d+) hours\s+(\d+) minutes\s+([0-9.]+) seconds")


def elapsed_blocks(text: str) -> list[float]:
    return [int(d) * 86400 + int(h) * 3600 + int(m) * 60 + float(s) for d, h, m, s in ELAPSED_RE.findall(text)]


def run_logged(command: list[str], cwd: Path, *, stdin: Path | None = None, stdout: Path | None = None, env=None):
    start = time.perf_counter()
    inp = stdin.open("r") if stdin else None
    out = stdout.open("w") if stdout else subprocess.DEVNULL
    try:
        cp = subprocess.run(command, cwd=cwd, stdin=inp, stdout=out,
                            stderr=subprocess.STDOUT if stdout else None,
                            env=env, check=False)
    finally:
        if inp is not None:
            inp.close()
        if stdout is not None:
            out.close()
    return cp.returncode, time.perf_counter() - start


def kabsch_row(source: np.ndarray, target: np.ndarray):
    source_c = source - source.mean(axis=0)
    target_c = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(source_c.T @ target_c)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = source_c @ rotation + target.mean(axis=0)
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rotation, rmsd


def rotate_hessian(hessian: np.ndarray, rotation_row: np.ndarray, natoms: int) -> np.ndarray:
    transform = np.kron(np.eye(natoms), rotation_row.T)
    rotated = transform @ hessian @ transform.T
    return 0.5 * (rotated + rotated.T)


def rewrite_nproc(text: str, nproc: int) -> str:
    pattern = re.compile(r"(?im)^%nproc(?:shared)?\s*=\s*\d+\s*$")
    replacement = f"%nprocshared={nproc}"
    return pattern.sub(replacement, text, count=1) if pattern.search(text) else replacement + "\n" + text


def write_checkpoint(path: Path, geom, nproc: int, mode: str) -> None:
    route = f"#P {METHOD} SP" if mode == "sp" else f"#P {METHOD} Guess=(Only,Save)"
    coords = "\n".join(coord_lines(geom.atomic_nums, geom.coords_ang))
    path.write_text(
        f"%chk=mlip_readfc_initial.chk\n%nprocshared={nproc}\n{route}\n\n"
        f"{mode} checkpoint for ML Hessian handoff\n\n{geom.charge} {geom.mult}\n{coords}\n\n"
    )


def write_optfreq(path: Path, nproc: int) -> None:
    path.write_text(
        f"%chk=mlip_readfc.chk\n%nprocshared={nproc}\n"
        f"#P {METHOD} opt(ts,readfc,noeigen,nomicro,maxcycles=150) freq Geom=AllCheck Guess=TCheck\n\n"
        "QM TS optimization with imported EquiformerV2 Hessian\n\n"
    )


def source_base(dataset_dir: str) -> Path:
    return ROOT / dataset_dir


def prediction_times(hessian_root: Path) -> tuple[dict[tuple[str, str], float], float]:
    with (hessian_root / "timings.csv").open() as handle:
        values = {(row["dataset"], row["reaction"]): float(row["inference_time_s"]) for row in csv.DictReader(handle)}
    metadata = json.loads((hessian_root / "metadata.json").read_text())
    return values, float(metadata["model_load_amortized_per_reaction_s"])


def copy_contract(source: Path, local_rxn: Path) -> None:
    local_rxn.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "ts_guess.xyz", local_rxn / "ts_guess.xyz")
    src_irc = source / "IRC"
    if src_irc.is_dir():
        shutil.copytree(src_irc, local_rxn / "IRC", dirs_exist_ok=True)


def execute(task: dict[str, str | int | float]):
    dataset = str(task["dataset"])
    dataset_dir = str(task["dataset_dir"])
    reaction = str(task["reaction"])
    workflow = str(task["workflow"])
    nproc = int(task["nproc"])
    local_root = Path(str(task["local_root"]))
    result_root = Path(str(task["result_root"]))
    hessian_root = Path(str(task["hessian_root"])) if task.get("hessian_root") else None
    inference_s = float(task.get("inference_s", 0.0))
    load_share_s = float(task.get("load_share_s", 0.0))
    source_rxn = source_base(dataset_dir) / "Gaussian_calcfc" / reaction
    local_rxn = local_root / dataset / reaction
    work = local_rxn / "TS+Freq"
    scratch = local_root / "gaussian_scratch" / dataset / reaction
    result_rxn = result_root / dataset_dir / workflow / reaction
    work.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update({"GAUSS_SCRDIR": str(scratch), "OMP_THREAD_LIMIT": str(nproc), "OMP_NUM_THREADS": str(nproc)})
    row: dict[str, object] = {"dataset": dataset, "reaction": reaction, "workflow": workflow, "cores": nproc,
                              "node": os.uname().nodename, "job_id": os.environ.get("SLURM_JOB_ID", "")}
    stage_in_start = time.perf_counter()
    copy_contract(source_rxn, local_rxn)
    row["stage_in_s"] = time.perf_counter() - stage_in_start
    workflow_start = time.perf_counter()
    try:
        if workflow in ("Gaussian_calcfc", "Gaussian_calcall"):
            source_gjf = source_base(dataset_dir) / workflow / reaction / "TS+Freq" / "ts+freq.gjf"
            gjf = work / "ts+freq.gjf"
            gjf.write_text(rewrite_nproc(source_gjf.read_text(errors="replace"), nproc))
            rc, gaussian_s = run_logged(["g16"], work, stdin=gjf, stdout=work / "ts+freq.log", env=env)
            row.update({"g16_exit": rc, "gaussian_wrapper_s": gaussian_s, "reason": "completed" if rc == 0 else "g16_failed"})
        else:
            if hessian_root is None:
                raise RuntimeError("OneShot phase missing Hessian root")
            geom = parse_ts_gjf(Path(str(task["source_gjf"])))
            mode = "sp" if workflow == "mlip_oneshot_SP_Readfc" else "guessonly_save"
            checkpoint_gjf = work / "mlip_readfc_initial.gjf"
            checkpoint_log = work / "mlip_readfc_initial.log"
            write_checkpoint(checkpoint_gjf, geom, nproc, mode)
            rc_checkpoint, checkpoint_s = run_logged(["g16"], work, stdin=checkpoint_gjf, stdout=checkpoint_log, env=env)
            row.update({"checkpoint_mode": mode, "checkpoint_exit": rc_checkpoint, "checkpoint_s": checkpoint_s,
                        "hessian_inference_s": inference_s, "model_load_amortized_s": load_share_s})
            if rc_checkpoint != 0:
                row["reason"] = "checkpoint_failed"
            else:
                rc_form, form_s = run_logged(["formchk", "-3", "mlip_readfc_initial.chk", "mlip_readfc_initial.fchk"], work, env=env)
                row.update({"formchk_exit": rc_form, "formchk_s": form_s})
                if rc_form != 0:
                    row["reason"] = "formchk_failed"
                else:
                    artifact = np.load(hessian_root / dataset / f"{reaction}.npz")
                    source_nums = artifact["atomic_numbers"].astype(np.int64)
                    source_coords = artifact["coordinates_angstrom"].astype(np.float64)
                    source_hessian = artifact["hessian_hartree_bohr2"].astype(np.float64)
                    fchk_nums = read_fchk_int_array(work / "mlip_readfc_initial.fchk", "Atomic numbers")
                    fchk_coords = read_fchk_array(work / "mlip_readfc_initial.fchk", "Current cartesian coordinates").reshape(-1, 3) * BOHR_TO_ANG
                    if not np.array_equal(fchk_nums, source_nums):
                        raise RuntimeError("fchk/prediction atom order mismatch")
                    rotation, alignment_rmsd = kabsch_row(source_coords, fchk_coords)
                    if alignment_rmsd > 1.0e-5:
                        raise RuntimeError(f"checkpoint coordinate RMSD {alignment_rmsd:.3e} A")
                    rotated_hessian = rotate_hessian(source_hessian, rotation, len(fchk_nums))
                    edit_start = time.perf_counter()
                    replace_or_insert_force_constants(work / "mlip_readfc_initial.fchk", work / "mlip_readfc.fchk", rotated_hessian)
                    edit_s = time.perf_counter() - edit_start
                    rc_unfchk, unfchk_s = run_logged(["unfchk", "mlip_readfc.fchk", "mlip_readfc.chk"], work, env=env)
                    row.update({"alignment_rmsd_ang": alignment_rmsd, "fchk_edit_s": edit_s,
                                "unfchk_exit": rc_unfchk, "unfchk_s": unfchk_s})
                    if rc_unfchk != 0:
                        row["reason"] = "unfchk_failed"
                    else:
                        gjf = work / "ts+freq.gjf"
                        write_optfreq(gjf, nproc)
                        rc, gaussian_s = run_logged(["g16"], work, stdin=gjf, stdout=work / "ts+freq.log", env=env)
                        row.update({"g16_exit": rc, "gaussian_wrapper_s": gaussian_s,
                                    "reason": "completed" if rc == 0 else "g16_failed"})
        log = work / "ts+freq.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        blocks = elapsed_blocks(text)
        row.update({
            "gaussian_elapsed_blocks_s": ";".join(f"{x:.3f}" for x in blocks),
            "stationary_found": int("Stationary point found" in text),
            "normal_termination_count": text.count("Normal termination of Gaussian 16"),
            "local_workflow_wall_s": time.perf_counter() - workflow_start,
        })
    except Exception as exc:
        row["reason"] = f"runner_error:{type(exc).__name__}:{exc}"
    stage_out_start = time.perf_counter()
    result_rxn.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_rxn, result_rxn, dirs_exist_ok=True)
    ts_dest = result_rxn / "TS+Freq"
    row["stage_out_s"] = time.perf_counter() - stage_out_start
    row["preparation_inclusive_wall_s"] = float(row.get("local_workflow_wall_s", 0.0)) + inference_s + load_share_s + float(row["stage_in_s"]) + float(row["stage_out_s"])
    timing_name = f"timing_{nproc}core.tsv"
    with (ts_dest / timing_name).open("w") as handle:
        for key in sorted(row):
            handle.write(f"{key}\t{row[key]}\n")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--hessians", type=Path)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open() as handle:
        entries = list(csv.DictReader(handle))
    args.local_root.mkdir(parents=True, exist_ok=False)
    args.stats.mkdir(parents=True, exist_ok=True)
    inference, load_share = prediction_times(args.hessians) if args.hessians else ({}, 0.0)
    tasks = []
    for entry in entries:
        task: dict[str, str | int | float] = dict(entry)
        task.update({"workflow": args.workflow, "nproc": args.nproc,
                     "local_root": str(args.local_root), "result_root": str(args.result_root),
                     "hessian_root": str(args.hessians) if args.hessians else "",
                     "inference_s": inference.get((entry["dataset"], entry["reaction"]), 0.0),
                     "load_share_s": load_share})
        tasks.append(task)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(execute, task): (task["dataset"], task["reaction"]) for task in tasks}
        for future in as_completed(futures):
            dataset, reaction = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"dataset": dataset, "reaction": reaction, "workflow": args.workflow,
                       "reason": f"worker_crash:{type(exc).__name__}:{exc}"}
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    fields = sorted({key for row in rows for key in row})
    with (args.stats / "timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: (str(x["dataset"]), str(x["reaction"]))))
    failures = [row for row in rows if str(row.get("reason", "")).startswith(("runner_error", "worker_crash"))]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
