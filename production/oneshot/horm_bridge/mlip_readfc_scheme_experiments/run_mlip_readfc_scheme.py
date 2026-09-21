#!/usr/bin/env python3
"""Run isolated MLIP ReadFC optimization experiments.

This runner intentionally writes to new per-reaction scheme directories and
does not modify existing TS+Freq or earlier MLIP result directories.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CALCALL = PROJECT / "calcall"
QM_METHOD = "wB97X/6-31G(d)"
PURE_EXTERNAL = PROJECT / "horm_mlip_pure.sh"
BOHR_TO_ANG = 0.529177210903

sys.path.insert(0, str(PROJECT))
from prepare_mlip_readfc import (  # noqa: E402
    MlipHessianComputer,
    parse_ts_gjf,
    read_fchk_array,
    read_fchk_int_array,
    replace_or_insert_force_constants,
)
from prepare_mlip_preopt_readfc import (  # noqa: E402
    check_ts_freq_log,
    coords_from_last_orientation,
    preopt_has_usable_geometry,
    selected_rxn_dirs,
    write_xyz,
)
from run_mlip_preopt_guarded_cpu_fchk_rehessian_readfc import (  # noqa: E402
    aligned_displacements,
)

Z_TO_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


@dataclass(frozen=True)
class Scheme:
    key: str
    outdir_name: str
    preopt_mode: str
    readfc_route: str
    max_preopt_steps: int = 0
    tiny_maxstep: int = 5
    mode_step_ang: float = 0.04
    hessian_quality_filter: bool = False


SCHEMES: dict[str, Scheme] = {
    "A": Scheme(
        key="A",
        outdir_name="SchemeA_AdaptivePreopt_GuardedCpuFchkReHessianReadFC",
        preopt_mode="adaptive",
        max_preopt_steps=5,
        readfc_route="opt(ts,readfc,noeigen,nomicro,maxcycles=150)",
    ),
    "B": Scheme(
        key="B",
        outdir_name="SchemeB_TinyStepPreopt_GuardedCpuFchkReHessianReadFC",
        preopt_mode="tiny_step",
        max_preopt_steps=5,
        tiny_maxstep=5,
        readfc_route="opt(ts,readfc,noeigen,nomicro,maxcycles=150)",
    ),
    "C": Scheme(
        key="C",
        outdir_name="SchemeC_ModeGuidedPreopt_GuardedCpuFchkReHessianReadFC",
        preopt_mode="mode_guided",
        max_preopt_steps=1,
        readfc_route="opt(ts,readfc,noeigen,nomicro,maxcycles=150)",
    ),
    "D": Scheme(
        key="D",
        outdir_name="SchemeD_MLIPHessianQualityFilter_ReadFC",
        preopt_mode="none",
        hessian_quality_filter=True,
        readfc_route="opt(ts,readfc,noeigen,nomicro,maxcycles=150)",
    ),
    "E": Scheme(
        key="E",
        outdir_name="SchemeE_ReadFC_RecalcFC5",
        preopt_mode="none",
        readfc_route="opt(ts,readfc,recalcfc=5,noeigen,nomicro,maxcycles=150)",
    ),
    "F": Scheme(
        key="F",
        outdir_name="SchemeF_ReadFC_RecalcFC10",
        preopt_mode="none",
        readfc_route="opt(ts,readfc,recalcfc=10,noeigen,nomicro,maxcycles=150)",
    ),
    "O": Scheme(
        key="O",
        outdir_name="SchemeMLIPOneShot_OriginalGeometry_ReadFC",
        preopt_mode="none",
        readfc_route="opt(ts,readfc,noeigen,nomicro,maxcycles=150)",
    ),
}


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


def run(cmd: list[str], cwd: Path) -> None:
    print(f"[cmd] ({cwd}) {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def source_geometry(rxn_dir: Path):
    candidates = [
        rxn_dir / "TS+Freq" / "mlip_readfc_initial.gjf",
        rxn_dir / "TS+Freq" / "pure_gaussian_calcfc_all.gjf",
        rxn_dir / "TS+Freq" / "ts+freq.gjf",
    ]
    errors: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            return parse_ts_gjf(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    detail = "; ".join(errors) if errors else "no candidate gjf files"
    raise FileNotFoundError(f"{rxn_dir.name}: no parseable source geometry ({detail})")


def coord_lines(atomic_nums: np.ndarray, coords_ang: np.ndarray) -> list[str]:
    lines = []
    for z, xyz in zip(atomic_nums, coords_ang):
        sym = Z_TO_SYMBOL.get(int(z), str(int(z)))
        lines.append(f"{sym:<2s} {xyz[0]:18.10f} {xyz[1]:18.10f} {xyz[2]:18.10f}")
    return lines


def write_mlip_preopt_gjf(
    outdir: Path,
    *,
    name: str,
    atomic_nums: np.ndarray,
    coords_ang: np.ndarray,
    charge: int,
    mult: int,
    maxcycles: int,
    maxstep: int | None = None,
) -> Path:
    path = outdir / f"{name}.gjf"
    maxstep_kw = f",maxstep={maxstep}" if maxstep is not None else ""
    coords = "\n".join(coord_lines(atomic_nums, coords_ang))
    text = f"""%chk={name}.chk
%nproc=1
#P opt(ts,calcall,noeigentest,nomicro,maxcycles={maxcycles}{maxstep_kw}) external='{PURE_EXTERNAL}'

Pure MLIP TS preopt for isolated ReadFC scheme

{charge} {mult}
{coords}

"""
    path.write_text(text, encoding="utf-8")
    return path


def write_sp_gjf(
    outdir: Path,
    *,
    atomic_nums: np.ndarray,
    coords_ang: np.ndarray,
    charge: int,
    mult: int,
    nproc: str,
    method: str,
) -> Path:
    path = outdir / "mlip_readfc_initial.gjf"
    coords = "\n".join(coord_lines(atomic_nums, coords_ang))
    text = f"""%chk=mlip_readfc_initial.chk
{nproc}
#P {method} SP

QM SP checkpoint for isolated MLIP ReadFC scheme

{charge} {mult}
{coords}

"""
    path.write_text(text, encoding="utf-8")
    return path


def write_readfc_gjf(outdir: Path, *, nproc: str, method: str, route: str) -> Path:
    path = outdir / "ts+freq.gjf"
    text = f"""%chk=mlip_readfc.chk
{nproc}
#P {method} {route} freq Geom=AllCheck Guess=TCheck

QM TS opt+freq for isolated MLIP ReadFC scheme

"""
    path.write_text(text, encoding="utf-8")
    return path


def mlip_predict(computer: MlipHessianComputer, atomic_nums: np.ndarray, coords_ang: np.ndarray, *, need_hessian: bool):
    if computer.horm is None:
        raise RuntimeError("MLIP model is not loaded")
    coords_bohr = coords_ang / BOHR_TO_ANG
    energy_ev, forces_ev_ang, hess_ev_ang2 = computer.horm._forward_predict(
        computer.pm,
        computer.model_name,
        computer.device,
        atomic_nums,
        coords_bohr,
        need_hessian=need_hessian,
    )
    hess = None
    if hess_ev_ang2 is not None:
        if hasattr(hess_ev_ang2, "detach"):
            hess_ev_ang2 = hess_ev_ang2.detach().cpu().numpy()
        hess_arr = np.asarray(hess_ev_ang2, dtype=np.float64) * computer.horm.HESS_CONV
        hess = 0.5 * (hess_arr + hess_arr.T)
    return float(energy_ev), np.asarray(forces_ev_ang, dtype=np.float64), hess


def run_single_preopt(
    outdir: Path,
    *,
    geom,
    atomic_nums: np.ndarray,
    coords_ang: np.ndarray,
    maxcycles: int,
    name: str,
    maxstep: int | None = None,
) -> tuple[int, np.ndarray]:
    gjf = write_mlip_preopt_gjf(
        outdir,
        name=name,
        atomic_nums=atomic_nums,
        coords_ang=coords_ang,
        charge=geom.charge,
        mult=geom.mult,
        maxcycles=maxcycles,
        maxstep=maxstep,
    )
    log = outdir / f"{name}.log"
    exit_code = run_g16_input(outdir, gjf, log)
    if exit_code != 0 and not preopt_has_usable_geometry(log):
        raise RuntimeError(f"MLIP preopt {name} failed with exit {exit_code}")
    got_nums, got_coords = coords_from_last_orientation(log, len(atomic_nums))
    if not np.array_equal(got_nums, atomic_nums):
        raise RuntimeError("atom order changed during MLIP preopt")
    return exit_code, got_coords


def choose_preopt_geometry(
    scheme: Scheme,
    outdir: Path,
    geom,
    computer: MlipHessianComputer,
    *,
    max_rmsd_ang: float,
    max_atom_displacement_ang: float,
    adaptive_min_step_rmsd_ang: float,
) -> dict[str, object]:
    original = np.asarray(geom.coords_ang, dtype=np.float64)
    atomic_nums = np.asarray(geom.atomic_nums, dtype=np.int64)
    selected = original.copy()
    raw = original.copy()
    exits: list[int] = []
    mode = scheme.preopt_mode
    reason = "no_preopt"

    if mode == "none":
        pass
    elif mode == "adaptive":
        current = original.copy()
        reason = "max_steps"
        for step in range(1, scheme.max_preopt_steps + 1):
            exit_code, trial = run_single_preopt(
                outdir,
                geom=geom,
                atomic_nums=atomic_nums,
                coords_ang=current,
                maxcycles=1,
                name=f"mlip_adaptive_preopt_step{step}",
            )
            exits.append(exit_code)
            raw = trial
            disp_total = aligned_displacements(original, trial)
            rms_total = float(np.sqrt(np.mean(disp_total**2)))
            max_total = float(np.max(disp_total))
            disp_step = aligned_displacements(current, trial)
            step_rms = float(np.sqrt(np.mean(disp_step**2)))
            if rms_total > max_rmsd_ang or max_total > max_atom_displacement_ang:
                reason = f"guard_stop_step{step}"
                break
            selected = trial
            current = trial
            if step_rms < adaptive_min_step_rmsd_ang:
                reason = f"small_step_stop_step{step}"
                break
    elif mode == "tiny_step":
        exit_code, trial = run_single_preopt(
            outdir,
            geom=geom,
            atomic_nums=atomic_nums,
            coords_ang=original,
            maxcycles=scheme.max_preopt_steps,
            maxstep=scheme.tiny_maxstep,
            name="mlip_tiny_step_preopt",
        )
        exits.append(exit_code)
        raw = trial
        disp = aligned_displacements(original, trial)
        if float(np.sqrt(np.mean(disp**2))) <= max_rmsd_ang and float(np.max(disp)) <= max_atom_displacement_ang:
            selected = trial
            reason = "tiny_step_selected"
        else:
            reason = "tiny_step_guard_fallback"
    elif mode == "mode_guided":
        energy0, _, hess = mlip_predict(computer, atomic_nums, original, need_hessian=True)
        if hess is None:
            raise RuntimeError("MLIP did not return Hessian for mode-guided preopt")
        eigvals, eigvecs = np.linalg.eigh(hess)
        mode_vec = eigvecs[:, int(np.argmin(eigvals))].reshape(len(atomic_nums), 3)
        mode_vec /= max(float(np.linalg.norm(mode_vec)), 1.0e-12)
        plus = original + scheme.mode_step_ang * mode_vec
        minus = original - scheme.mode_step_ang * mode_vec
        e_plus, _, _ = mlip_predict(computer, atomic_nums, plus, need_hessian=False)
        e_minus, _, _ = mlip_predict(computer, atomic_nums, minus, need_hessian=False)
        raw = plus if e_plus <= e_minus else minus
        disp = aligned_displacements(original, raw)
        if float(np.sqrt(np.mean(disp**2))) <= max_rmsd_ang and float(np.max(disp)) <= max_atom_displacement_ang:
            selected = raw
            reason = "mode_guided_selected"
        else:
            reason = "mode_guided_guard_fallback"
        (outdir / "mode_guided_metadata.json").write_text(
            json.dumps(
                {
                    "energy_original_ev": energy0,
                    "energy_plus_ev": e_plus,
                    "energy_minus_ev": e_minus,
                    "lowest_cartesian_hessian_eigenvalue": float(np.min(eigvals)),
                    "mode_step_ang": scheme.mode_step_ang,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        raise ValueError(f"unknown preopt_mode: {mode}")

    disp = aligned_displacements(original, selected)
    raw_disp = aligned_displacements(original, raw)
    write_xyz(outdir / "raw_preopt_geom.xyz", atomic_nums, raw)
    write_xyz(outdir / "preopt_geom.xyz", atomic_nums, selected)
    return {
        "coords_ang": selected,
        "selected_geometry_source": "original_geometry_fallback" if np.allclose(selected, original) else "mlip_preopt",
        "preopt_stop_reason": reason,
        "preopt_exits": exits,
        "preopt_rmsd_ang": float(np.sqrt(np.mean(disp**2))),
        "preopt_max_disp_ang": float(np.max(disp)),
        "raw_preopt_rmsd_ang": float(np.sqrt(np.mean(raw_disp**2))),
        "raw_preopt_max_disp_ang": float(np.max(raw_disp)),
    }


def hessian_quality(hess: np.ndarray) -> dict[str, object]:
    hess = np.asarray(hess, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(0.5 * (hess + hess.T))
    abs_e = np.abs(eigvals)
    finite = bool(np.all(np.isfinite(eigvals)))
    max_abs = float(np.max(abs_e)) if eigvals.size else 0.0
    min_nonzero = float(np.min(abs_e[abs_e > 1.0e-10])) if np.any(abs_e > 1.0e-10) else 0.0
    cond = max_abs / min_nonzero if min_nonzero > 0 else float("inf")
    strong_negative = int(np.sum(eigvals < -1.0e-4))
    ok = finite and max_abs < 100.0 and cond < 1.0e12 and strong_negative >= 1
    return {
        "ok": bool(ok),
        "finite": finite,
        "min_eig": float(np.min(eigvals)) if eigvals.size else 0.0,
        "max_eig": float(np.max(eigvals)) if eigvals.size else 0.0,
        "max_abs_eig": max_abs,
        "condition_estimate": float(cond),
        "strong_negative_cartesian_eigs": strong_negative,
    }


def prepare_one(
    rxn_dir: Path,
    scheme: Scheme,
    computer: MlipHessianComputer,
    *,
    nproc: str,
    method: str,
    force: bool,
    max_rmsd_ang: float,
    max_atom_displacement_ang: float,
    adaptive_min_step_rmsd_ang: float,
) -> tuple[Path, dict[str, object]]:
    geom = source_geometry(rxn_dir)
    outdir = rxn_dir / scheme.outdir_name
    if outdir.exists():
        if force:
            shutil.rmtree(outdir)
        else:
            return outdir, {"prep_status": "exists_skipped"}
    outdir.mkdir(parents=True, exist_ok=False)

    pre = choose_preopt_geometry(
        scheme,
        outdir,
        geom,
        computer,
        max_rmsd_ang=max_rmsd_ang,
        max_atom_displacement_ang=max_atom_displacement_ang,
        adaptive_min_step_rmsd_ang=adaptive_min_step_rmsd_ang,
    )
    selected_coords = np.asarray(pre["coords_ang"], dtype=np.float64)
    atomic_nums = np.asarray(geom.atomic_nums, dtype=np.int64)

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
    natoms = len(atomic_nums)
    if coords_bohr.size != 3 * natoms:
        raise RuntimeError(f"{rxn_dir.name}: bad fchk coordinate count {coords_bohr.size}")
    fchk_coords_ang = coords_bohr.reshape(natoms, 3) * BOHR_TO_ANG
    hess = np.asarray(
        computer.hessian_hartree_bohr2(fchk_atomic_nums, fchk_coords_ang),
        dtype=np.float64,
    )
    q = hessian_quality(hess)
    hess_path = outdir / "mlip_hessian_cpu_recomputed_on_fchk_hartree_bohr2.npy"
    np.save(hess_path, np.asarray(hess, dtype=np.float64))

    route = scheme.readfc_route
    hessian_used = True
    if scheme.hessian_quality_filter and not q["ok"]:
        route = "opt(ts,calcfc,noeigen,nomicro,maxcycles=150)"
        hessian_used = False
        shutil.copyfile(initial_fchk, modified_fchk)
        shutil.copyfile(initial_chk, modified_chk)
    else:
        replace_or_insert_force_constants(initial_fchk, modified_fchk, hess)
        modified_chk.unlink(missing_ok=True)
        run(["unfchk", modified_fchk.name, modified_chk.name], outdir)
    write_readfc_gjf(outdir, nproc=nproc, method=method, route=route)

    metadata = {
        "stem": rxn_dir.name,
        "scheme_key": scheme.key,
        "scheme_outdir": scheme.outdir_name,
        "preopt_mode": scheme.preopt_mode,
        "readfc_route": route,
        "hessian_used": hessian_used,
        "hessian_quality_filter": scheme.hessian_quality_filter,
        "hessian_quality": q,
        "hessian_source": "cpu_recomputed_on_gaussian_fchk_coordinates",
        "hessian_file": hess_path.name,
        "charge": geom.charge,
        "mult": geom.mult,
        "atomic_nums": atomic_nums.astype(int).tolist(),
        "max_rmsd_ang": max_rmsd_ang,
        "max_atom_displacement_ang": max_atom_displacement_ang,
        **{k: v for k, v in pre.items() if k != "coords_ang"},
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return outdir, {
        "prep_status": "ok",
        "sp_exit": sp_exit,
        "geom_source": metadata["selected_geometry_source"],
        "preopt_rmsd_ang": f"{metadata['preopt_rmsd_ang']:.6f}",
        "preopt_max_disp_ang": f"{metadata['preopt_max_disp_ang']:.6f}",
        "hessian_filter_ok": int(bool(q["ok"])),
        "hessian_used": int(hessian_used),
        "route": route,
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


def parse_args(default_scheme: str | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default=default_scheme, required=default_scheme is None, choices=sorted(SCHEMES))
    ap.add_argument("--outdir-name", default="", help="override the default per-rxn scheme output directory")
    ap.add_argument("--calcall-root", type=Path, default=CALCALL)
    ap.add_argument("--rxn", action="append", default=[])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--method", default=QM_METHOD)
    ap.add_argument("--nproc", default="%nproc=24")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run-ts-freq", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--max-rmsd-ang", type=float, default=0.08)
    ap.add_argument("--max-atom-displacement-ang", type=float, default=0.20)
    ap.add_argument("--adaptive-min-step-rmsd-ang", type=float, default=0.005)
    ap.add_argument("--run-stats-dir", type=Path, default=PROJECT / "slurm_logs" / "run_stats")
    ap.add_argument("--job-tag", default=os.environ.get("SLURM_JOB_ID", "manual"))
    return ap.parse_args()


def main_with_scheme(default_scheme: str | None = None) -> int:
    args = parse_args(default_scheme)
    scheme = SCHEMES[args.scheme]
    if args.outdir_name:
        scheme = Scheme(
            key=scheme.key,
            outdir_name=args.outdir_name,
            preopt_mode=scheme.preopt_mode,
            readfc_route=scheme.readfc_route,
            max_preopt_steps=scheme.max_preopt_steps,
            tiny_maxstep=scheme.tiny_maxstep,
            mode_step_ang=scheme.mode_step_ang,
            hessian_quality_filter=scheme.hessian_quality_filter,
        )
    dirs = selected_rxn_dirs(args.calcall_root.resolve(), args.rxn, args.limit)
    if not dirs:
        raise SystemExit("no rxn directories selected")
    for exe in ("g16", "formchk", "unfchk"):
        if not shutil.which(exe):
            raise SystemExit(f"{exe} not found in PATH")
    if not PURE_EXTERNAL.is_file() and scheme.preopt_mode in {"adaptive", "tiny_step"}:
        raise SystemExit(f"missing pure MLIP external: {PURE_EXTERNAL}")

    args.run_stats_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.run_stats_dir / f"{scheme.outdir_name}_{args.job_tag}.csv"
    check_py = PROJECT / "slurm_logs" / "check_ts_freq_log.py"
    fields = [
        "stem",
        "scheme",
        "outdir",
        "prep_status",
        "sp_exit",
        "geom_source",
        "preopt_rmsd_ang",
        "preopt_max_disp_ang",
        "hessian_filter_ok",
        "hessian_used",
        "route",
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
                outdir = rxn_dir / scheme.outdir_name
                row = {
                    "stem": rxn_dir.name,
                    "scheme": scheme.outdir_name,
                    "outdir": str(outdir),
                    "prep_status": "",
                    "sp_exit": "",
                    "geom_source": "",
                    "preopt_rmsd_ang": "",
                    "preopt_max_disp_ang": "",
                    "hessian_filter_ok": "",
                    "hessian_used": "",
                    "route": "",
                    "g16_exit": "",
                    "strict_ok": "0",
                    "reason": "",
                    "elapsed_s": 0,
                }
                try:
                    outdir, prep_row = prepare_one(
                        rxn_dir,
                        scheme,
                        computer,
                        nproc=args.nproc,
                        method=args.method,
                        force=args.force,
                        max_rmsd_ang=args.max_rmsd_ang,
                        max_atom_displacement_ang=args.max_atom_displacement_ang,
                        adaptive_min_step_rmsd_ang=args.adaptive_min_step_rmsd_ang,
                    )
                    row["outdir"] = str(outdir)
                    row.update(prep_row)
                    if args.run_ts_freq and row["prep_status"] == "ok":
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
        "scheme_key": scheme.key,
        "scheme": scheme.outdir_name,
        "total": len(dirs),
        "g16_runs": run_count,
        "strict_ok": ok_count,
        "csv": str(csv_path),
    }
    summary_path = args.run_stats_dir / f"{scheme.outdir_name}_{args.job_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_with_scheme())
