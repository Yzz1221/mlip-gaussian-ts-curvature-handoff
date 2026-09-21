#!/usr/bin/env python3
"""Isolated experiment: pure-MLIP short TS preopt, then QM readfc TS+freq.

Outputs go to calcall/rxn*/MLIP_Preopt{N}_ReadFC/ and do not overwrite the
existing TS+Freq, pure-Gaussian, or hybrid-calcall result directories.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
    parse_ts_gjf,
    read_fchk_array,
    read_fchk_int_array,
    replace_or_insert_force_constants,
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


def run(cmd: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"[cmd] ({cwd}) {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, cwd=str(cwd), check=check)


def selected_rxn_dirs(root: Path, rxns: list[str], limit: int | None) -> list[Path]:
    if rxns:
        dirs = [root / x for x in rxns]
    else:
        dirs = sorted(
            [p for p in root.iterdir() if p.is_dir() and p.name.startswith("rxn")],
            key=lambda p: int(p.name[3:]) if p.name[3:].isdigit() else p.name,
        )
    return dirs if limit is None else dirs[:limit]


def source_gjf_candidates(rxn_dir: Path) -> list[Path]:
    candidates = [
        rxn_dir / "TS+Freq" / "ts+freq.gjf",
        rxn_dir / "TS+Freq" / "mlip_readfc_initial.gjf",
    ]
    return [path for path in candidates if path.is_file()]


def source_geometry(rxn_dir: Path):
    errors: list[str] = []
    for path in source_gjf_candidates(rxn_dir):
        try:
            return parse_ts_gjf(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    detail = "; ".join(errors) if errors else "no candidate gjf files"
    raise FileNotFoundError(f"{rxn_dir.name}: no parseable source TS geometry ({detail})")


def write_mlip_preopt_gjf(
    outdir: Path,
    geom,
    *,
    nproc: str,
    maxcycles: int,
    external: Path,
) -> Path:
    coords = "\n".join(geom.coord_lines)
    path = outdir / "mlip_preopt.gjf"
    text = f"""%chk=mlip_preopt.chk
{nproc}
#P opt(ts,calcall,noeigentest,nomicro,maxcycles={maxcycles}) external='{external}'

Pure MLIP TS preopt {maxcycles} steps

{geom.charge} {geom.mult}
{coords}

"""
    path.write_text(text, encoding="utf-8")
    return path


def coords_from_preopt_chk(outdir: Path, natoms: int) -> tuple[np.ndarray, np.ndarray]:
    chk = outdir / "mlip_preopt.chk"
    fchk = outdir / "mlip_preopt.fchk"
    if not chk.is_file():
        raise RuntimeError(f"preopt did not create {chk}")
    run(["formchk", "-3", chk.name, fchk.name], outdir)
    atomic_nums = read_fchk_int_array(fchk, "Atomic numbers")
    coords_bohr = read_fchk_array(fchk, "Current cartesian coordinates")
    if len(atomic_nums) != natoms or coords_bohr.size != 3 * natoms:
        raise RuntimeError(
            f"preopt fchk geometry size mismatch: atoms={len(atomic_nums)} coords={coords_bohr.size}"
        )
    return atomic_nums, coords_bohr.reshape(natoms, 3) * BOHR_TO_ANG


def coords_from_last_orientation(log: Path, natoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Return atomic numbers and coordinates from the last Gaussian orientation block."""
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [
        i
        for i, ln in enumerate(lines)
        if "Standard orientation:" in ln or "Input orientation:" in ln
    ]
    for start in reversed(starts):
        header = None
        for i in range(start, min(start + 12, len(lines))):
            if "Center" in lines[i] and "Atomic" in lines[i] and "Coordinates" in lines[i]:
                header = i
                break
        if header is None:
            continue
        idx = header + 1
        while idx < len(lines) and not (lines[idx].strip() and set(lines[idx].strip()) <= {"-"}):
            idx += 1
        while idx < len(lines) and set(lines[idx].strip()) <= {"-"}:
            idx += 1
        nums: list[int] = []
        coords: list[list[float]] = []
        while idx < len(lines):
            s = lines[idx].strip()
            if not s or set(s) <= {"-"}:
                break
            parts = s.split()
            if len(parts) < 6:
                break
            try:
                nums.append(int(parts[1]))
                coords.append([float(parts[3]), float(parts[4]), float(parts[5])])
            except ValueError:
                break
            idx += 1
        if len(nums) == natoms:
            return np.asarray(nums, dtype=np.int64), np.asarray(coords, dtype=np.float64)
    raise RuntimeError(f"{log}: no complete orientation block with {natoms} atoms")


def preopt_has_usable_geometry(log: Path) -> bool:
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    return (
        "Optimization stopped." in text
        and "-- Number of steps exceeded" in text
        and "Error termination request processed by link 9999" in text
    )


def write_xyz(path: Path, atomic_nums: np.ndarray, coords_ang: np.ndarray) -> None:
    lines = [str(len(atomic_nums)), "MLIP preoptimized geometry"]
    for z, xyz in zip(atomic_nums, coords_ang):
        sym = Z_TO_SYMBOL.get(int(z), str(int(z)))
        lines.append(f"{sym:2s} {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def coord_lines(atomic_nums: np.ndarray, coords_ang: np.ndarray) -> list[str]:
    out = []
    for z, xyz in zip(atomic_nums, coords_ang):
        sym = Z_TO_SYMBOL.get(int(z), str(int(z)))
        out.append(f"{sym:<2s} {xyz[0]:18.10f} {xyz[1]:18.10f} {xyz[2]:18.10f}")
    return out


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

QM SP checkpoint after pure MLIP preopt

{charge} {mult}
{coords}

"""
    path.write_text(text, encoding="utf-8")
    return path


def write_readfc_gjf(outdir: Path, *, nproc: str, method: str) -> Path:
    path = outdir / "ts+freq.gjf"
    text = f"""%chk=mlip_readfc.chk
{nproc}
#P {method} opt(ts,readfc,noeigen,nomicro,maxcycles=150) freq Geom=AllCheck Guess=TCheck

QM TS opt+freq after pure MLIP preopt and MLIP readfc

"""
    path.write_text(text, encoding="utf-8")
    return path


def check_ts_freq_log(check_py: Path, log: Path) -> tuple[str, str]:
    if not check_py.is_file():
        return "0", "missing_check_script"
    cp = subprocess.run(
        [sys.executable, str(check_py), str(log)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    parts = cp.stdout.strip().split("\t", 1)
    if not parts or not parts[0]:
        return "0", cp.stderr.strip().replace(",", ";") or "check_failed"
    return parts[0], parts[1].replace(",", ";") if len(parts) > 1 else ""


def prepare_one(
    rxn_dir: Path,
    computer: MlipHessianComputer,
    *,
    outdir_name: str,
    nproc: str,
    method: str,
    preopt_cycles: int,
    force: bool,
) -> tuple[str, int]:
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
    with preopt_gjf.open("r", encoding="utf-8") as inp, preopt_log.open(
        "w", encoding="utf-8"
    ) as out:
        cp = subprocess.run(
            ["g16"],
            cwd=str(outdir),
            stdin=inp,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if cp.returncode != 0 and not preopt_has_usable_geometry(preopt_log):
        raise RuntimeError(f"MLIP preopt g16 failed with exit {cp.returncode}")

    natoms = len(geom.atomic_nums)
    atomic_nums, preopt_coords = coords_from_last_orientation(preopt_log, natoms)
    if not np.array_equal(atomic_nums, geom.atomic_nums):
        raise RuntimeError(f"{rxn_dir.name}: atom order changed during preopt")
    write_xyz(outdir / "preopt_geom.xyz", atomic_nums, preopt_coords)

    sp_gjf = write_sp_gjf(
        outdir,
        atomic_nums=atomic_nums,
        coords_ang=preopt_coords,
        charge=geom.charge,
        mult=geom.mult,
        nproc=nproc,
        method=method,
    )
    run(["g16", sp_gjf.name], outdir)
    initial_chk = outdir / "mlip_readfc_initial.chk"
    initial_fchk = outdir / "mlip_readfc_initial.fchk"
    modified_fchk = outdir / "mlip_readfc.fchk"
    modified_chk = outdir / "mlip_readfc.chk"
    run(["formchk", "-3", initial_chk.name, initial_fchk.name], outdir)

    fchk_atomic_nums = read_fchk_int_array(initial_fchk, "Atomic numbers")
    if not np.array_equal(fchk_atomic_nums, atomic_nums):
        raise RuntimeError(f"{rxn_dir.name}: atom order changed between preopt and SP")
    coords_bohr = read_fchk_array(initial_fchk, "Current cartesian coordinates")
    coords_ang = coords_bohr.reshape(natoms, 3) * BOHR_TO_ANG
    hess = computer.hessian_hartree_bohr2(fchk_atomic_nums, coords_ang)
    replace_or_insert_force_constants(initial_fchk, modified_fchk, hess)
    modified_chk.unlink(missing_ok=True)
    run(["unfchk", modified_fchk.name, modified_chk.name], outdir)
    write_readfc_gjf(outdir, nproc=nproc, method=method)
    return "ok", cp.returncode


def run_ts_freq(outdir: Path, check_py: Path) -> tuple[int, str, str, int]:
    log = outdir / "ts+freq.log"
    gjf = outdir / "ts+freq.gjf"
    start = int(subprocess.check_output(["date", "+%s"], text=True).strip())
    with gjf.open("r", encoding="utf-8") as inp, log.open("w", encoding="utf-8") as out:
        cp = subprocess.run(
            ["g16"],
            cwd=str(outdir),
            stdin=inp,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = int(subprocess.check_output(["date", "+%s"], text=True).strip()) - start
    strict_ok, reason = check_ts_freq_log(check_py, log)
    stamp = outdir / ".ts_freq_strict_ok"
    if strict_ok == "1":
        stamp.touch()
    else:
        stamp.unlink(missing_ok=True)
    return cp.returncode, strict_ok, reason, elapsed


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
    ap.add_argument("--run-stats-dir", type=Path, default=HERE / "slurm_logs" / "run_stats")
    ap.add_argument("--job-tag", default=os.environ.get("SLURM_JOB_ID", "manual"))
    args = ap.parse_args()

    outdir_name = args.outdir_name or f"MLIP_Preopt{args.preopt_cycles}_ReadFC"
    root = args.calcall_root.resolve()
    dirs = selected_rxn_dirs(root, args.rxn, args.limit)
    if not dirs:
        raise SystemExit("no rxn directories selected")

    args.run_stats_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.run_stats_dir / f"mlip_preopt{args.preopt_cycles}_readfc_{args.job_tag}.csv"
    check_py = HERE / "slurm_logs" / "check_ts_freq_log.py"
    fields = [
        "stem",
        "outdir",
        "prep_status",
        "preopt_exit",
        "g16_exit",
        "strict_ok",
        "reason",
        "elapsed_s",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        ok_count = run_count = 0
        with MlipHessianComputer() as computer:
            for rxn_dir in dirs:
                outdir = rxn_dir / outdir_name
                row = {
                    "stem": rxn_dir.name,
                    "outdir": str(outdir),
                    "prep_status": "",
                    "preopt_exit": "",
                    "g16_exit": "",
                    "strict_ok": "0",
                    "reason": "",
                    "elapsed_s": 0,
                }
                try:
                    status, preopt_exit = prepare_one(
                        rxn_dir,
                        computer,
                        outdir_name=outdir_name,
                        nproc=args.nproc,
                        method=args.method,
                        preopt_cycles=args.preopt_cycles,
                        force=args.force,
                    )
                    row["prep_status"] = status
                    row["preopt_exit"] = preopt_exit
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
    summary_path = args.run_stats_dir / f"mlip_preopt{args.preopt_cycles}_readfc_{args.job_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
