#!/usr/bin/env python3
"""Single-job guarded MLIP preopt + fchk-coordinate ReadFC workflow.

This is the no CPU/GPU ping-pong version:
  1. run short pure-MLIP TS preopt in the same CPU job,
  2. accept the preopt geometry only if it stays close to the original TS,
  3. run Gaussian SP/formchk at the selected geometry,
  4. recompute the MLIP Hessian on the exact Gaussian fchk coordinates,
  5. inject that Hessian into the same fchk/chk,
  6. run Gaussian ReadFC opt+freq.

Outputs are isolated under a new outdir and existing results are not touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CALCALL = HERE / "calcall"
QM_METHOD = "wB97X/6-31G(d)"
PURE_EXTERNAL = HERE / "horm_mlip_pure.sh"
BOHR_TO_ANG = 0.529177210903

sys.path.insert(0, str(HERE))
from prepare_mlip_readfc import (  # noqa: E402
    MlipHessianComputer,
    read_fchk_array,
    read_fchk_int_array,
    replace_or_insert_force_constants,
)
from prepare_mlip_preopt_readfc import (  # noqa: E402
    check_ts_freq_log,
    coords_from_last_orientation,
    preopt_has_usable_geometry,
    selected_rxn_dirs,
    source_geometry,
    write_mlip_preopt_gjf,
    write_readfc_gjf,
    write_sp_gjf,
    write_xyz,
    run,
)


def aligned_displacements(ref: np.ndarray, trial: np.ndarray) -> np.ndarray:
    if ref.shape != trial.shape:
        raise ValueError(f"geometry shape mismatch: {ref.shape} vs {trial.shape}")
    ref_c = ref - ref.mean(axis=0)
    trial_c = trial - trial.mean(axis=0)
    cov = trial_c.T @ ref_c
    u, _, vt = np.linalg.svd(cov)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return np.linalg.norm(trial_c @ rot - ref_c, axis=1)


def run_g16_input(outdir: Path, gjf: Path, log: Path | None = None) -> int:
    if log is None:
        return subprocess.run(["g16", gjf.name], cwd=str(outdir), check=False).returncode
    with gjf.open("r", encoding="utf-8") as inp, log.open("w", encoding="utf-8") as out:
        cp = subprocess.run(
            ["g16"],
            cwd=str(outdir),
            stdin=inp,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return cp.returncode


def prepare_one(
    rxn_dir: Path,
    computer: MlipHessianComputer,
    *,
    outdir_name: str,
    nproc: str,
    method: str,
    preopt_cycles: int,
    force: bool,
    max_rmsd_ang: float,
    max_atom_displacement_ang: float,
) -> dict[str, object]:
    geom = source_geometry(rxn_dir)
    outdir = rxn_dir / outdir_name
    if force and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not PURE_EXTERNAL.is_file():
        raise RuntimeError(f"missing pure MLIP external: {PURE_EXTERNAL}")

    preopt_gjf = write_mlip_preopt_gjf(
        outdir,
        geom,
        nproc="%nproc=1",
        maxcycles=preopt_cycles,
        external=PURE_EXTERNAL,
    )
    preopt_log = outdir / "mlip_preopt.log"
    preopt_exit = run_g16_input(outdir, preopt_gjf, preopt_log)
    if preopt_exit != 0 and not preopt_has_usable_geometry(preopt_log):
        raise RuntimeError(f"MLIP preopt g16 failed with exit {preopt_exit}")

    natoms = len(geom.atomic_nums)
    atomic_nums, preopt_coords = coords_from_last_orientation(preopt_log, natoms)
    if not np.array_equal(atomic_nums, geom.atomic_nums):
        raise RuntimeError(f"{rxn_dir.name}: atom order changed during preopt")

    disp = aligned_displacements(geom.coords_ang, preopt_coords)
    preopt_rmsd = float(np.sqrt(np.mean(disp**2)))
    preopt_max_disp = float(np.max(disp))
    use_preopt = True
    if max_rmsd_ang > 0 and preopt_rmsd > max_rmsd_ang:
        use_preopt = False
    if max_atom_displacement_ang > 0 and preopt_max_disp > max_atom_displacement_ang:
        use_preopt = False

    selected_coords = preopt_coords if use_preopt else geom.coords_ang
    geom_source = "mlip_preopt" if use_preopt else "original_geometry_fallback"
    write_xyz(outdir / "raw_preopt_geom.xyz", atomic_nums, preopt_coords)
    write_xyz(outdir / "preopt_geom.xyz", atomic_nums, selected_coords)

    sp_gjf = write_sp_gjf(
        outdir,
        atomic_nums=atomic_nums,
        coords_ang=selected_coords,
        charge=geom.charge,
        mult=geom.mult,
        nproc=nproc,
        method=method,
    )
    sp_exit = run_g16_input(outdir, sp_gjf)
    if sp_exit != 0:
        raise RuntimeError(f"{rxn_dir.name}: SP g16 failed with exit {sp_exit}")

    initial_chk = outdir / "mlip_readfc_initial.chk"
    initial_fchk = outdir / "mlip_readfc_initial.fchk"
    modified_fchk = outdir / "mlip_readfc.fchk"
    modified_chk = outdir / "mlip_readfc.chk"
    run(["formchk", "-3", initial_chk.name, initial_fchk.name], outdir)
    fchk_atomic_nums = read_fchk_int_array(initial_fchk, "Atomic numbers")
    if not np.array_equal(fchk_atomic_nums, atomic_nums):
        raise RuntimeError(f"{rxn_dir.name}: atom order changed in SP fchk")

    coords_bohr = read_fchk_array(initial_fchk, "Current cartesian coordinates")
    dim = 3 * natoms
    if coords_bohr.size != dim:
        raise RuntimeError(f"{rxn_dir.name}: expected {dim} fchk coords, got {coords_bohr.size}")
    fchk_coords_ang = coords_bohr.reshape(natoms, 3) * BOHR_TO_ANG
    hess = computer.hessian_hartree_bohr2(fchk_atomic_nums, fchk_coords_ang)
    hess_path = outdir / "mlip_hessian_cpu_recomputed_on_fchk_hartree_bohr2.npy"
    np.save(hess_path, np.asarray(hess, dtype=np.float64))
    replace_or_insert_force_constants(initial_fchk, modified_fchk, hess)
    modified_chk.unlink(missing_ok=True)
    run(["unfchk", modified_fchk.name, modified_chk.name], outdir)
    write_readfc_gjf(outdir, nproc=nproc, method=method)

    metadata = {
        "stem": rxn_dir.name,
        "workflow": "guarded_cpu_fchk_rehessian_readfc",
        "preopt_cycles": preopt_cycles,
        "selected_geometry_source": geom_source,
        "preopt_rmsd_ang": preopt_rmsd,
        "preopt_max_disp_ang": preopt_max_disp,
        "max_rmsd_ang": max_rmsd_ang,
        "max_atom_displacement_ang": max_atom_displacement_ang,
        "charge": geom.charge,
        "mult": geom.mult,
        "atomic_nums": atomic_nums.astype(int).tolist(),
        "hessian_source": "cpu_recomputed_on_gaussian_fchk_coordinates",
        "hessian_file": hess_path.name,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "prep_status": "ok",
        "preopt_exit": preopt_exit,
        "sp_exit": sp_exit,
        "geom_source": geom_source,
        "preopt_rmsd_ang": f"{preopt_rmsd:.6f}",
        "preopt_max_disp_ang": f"{preopt_max_disp:.6f}",
    }


def run_ts_freq(outdir: Path, check_py: Path) -> tuple[int, str, str, int]:
    gjf = outdir / "ts+freq.gjf"
    log = outdir / "ts+freq.log"
    start = int(subprocess.check_output(["date", "+%s"], text=True).strip())
    g16_exit = run_g16_input(outdir, gjf, log)
    elapsed = int(subprocess.check_output(["date", "+%s"], text=True).strip()) - start
    strict_ok, reason = check_ts_freq_log(check_py, log)
    stamp = outdir / ".ts_freq_strict_ok"
    if strict_ok == "1":
        stamp.touch()
    else:
        stamp.unlink(missing_ok=True)
    return g16_exit, strict_ok, reason, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calcall-root", type=Path, default=CALCALL)
    ap.add_argument("--rxn", action="append", default=[])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--preopt-cycles", type=int, default=3)
    ap.add_argument("--outdir-name", default="")
    ap.add_argument("--method", default=QM_METHOD)
    ap.add_argument("--nproc", default="%nproc=96")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run-ts-freq", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--max-rmsd-ang", type=float, default=0.08)
    ap.add_argument("--max-atom-displacement-ang", type=float, default=0.20)
    ap.add_argument("--run-stats-dir", type=Path, default=HERE / "slurm_logs" / "run_stats")
    ap.add_argument("--job-tag", default=os.environ.get("SLURM_JOB_ID", "manual"))
    args = ap.parse_args()

    outdir_name = args.outdir_name or f"MLIP_Preopt{args.preopt_cycles}_GuardedCpuFchkReHessianReadFC"
    dirs = selected_rxn_dirs(args.calcall_root.resolve(), args.rxn, args.limit)
    if not dirs:
        raise SystemExit("no rxn directories selected")

    args.run_stats_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.run_stats_dir / f"mlip_preopt{args.preopt_cycles}_guarded_cpu_fchk_rehessian_{args.job_tag}.csv"
    check_py = HERE / "slurm_logs" / "check_ts_freq_log.py"
    fields = [
        "stem",
        "outdir",
        "prep_status",
        "preopt_exit",
        "sp_exit",
        "geom_source",
        "preopt_rmsd_ang",
        "preopt_max_disp_ang",
        "g16_exit",
        "strict_ok",
        "reason",
        "elapsed_s",
    ]
    ok_count = run_count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        with MlipHessianComputer() as computer:
            for rxn_dir in dirs:
                outdir = rxn_dir / outdir_name
                row = {
                    "stem": rxn_dir.name,
                    "outdir": str(outdir),
                    "prep_status": "",
                    "preopt_exit": "",
                    "sp_exit": "",
                    "geom_source": "",
                    "preopt_rmsd_ang": "",
                    "preopt_max_disp_ang": "",
                    "g16_exit": "",
                    "strict_ok": "0",
                    "reason": "",
                    "elapsed_s": 0,
                }
                try:
                    row.update(
                        prepare_one(
                            rxn_dir,
                            computer,
                            outdir_name=outdir_name,
                            nproc=args.nproc,
                            method=args.method,
                            preopt_cycles=args.preopt_cycles,
                            force=args.force,
                            max_rmsd_ang=args.max_rmsd_ang,
                            max_atom_displacement_ang=args.max_atom_displacement_ang,
                        )
                    )
                    if args.run_ts_freq:
                        g16_exit, strict_ok, reason, elapsed = run_ts_freq(outdir, check_py)
                        row.update(
                            {
                                "g16_exit": g16_exit,
                                "strict_ok": strict_ok,
                                "reason": reason,
                                "elapsed_s": elapsed,
                            }
                        )
                        run_count += 1
                        ok_count += int(strict_ok == "1")
                except Exception as exc:
                    row["prep_status"] = f"prep_error:{type(exc).__name__}"
                    row["reason"] = str(exc).replace(",", ";")
                    print(f"[{rxn_dir.name}] error: {exc}", file=sys.stderr)
                    if not args.continue_on_error:
                        writer.writerow(row)
                        fh.flush()
                        raise
                writer.writerow(row)
                fh.flush()
                print(f"[{rxn_dir.name}] {row}", file=sys.stderr)

    summary = {
        "job_tag": args.job_tag,
        "preopt_cycles": args.preopt_cycles,
        "outdir_name": outdir_name,
        "total": len(dirs),
        "g16_runs": run_count,
        "strict_ok": ok_count,
        "csv": str(csv_path),
    }
    summary_path = args.run_stats_dir / f"mlip_preopt{args.preopt_cycles}_guarded_cpu_fchk_rehessian_{args.job_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
