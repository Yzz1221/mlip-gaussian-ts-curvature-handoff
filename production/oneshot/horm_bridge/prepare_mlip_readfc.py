#!/usr/bin/env python3
"""Prepare Gaussian readfc TS jobs with one-shot MLIP Hessians.

For each calcall/rxn*/TS+Freq/ts+freq.gjf this script can:
  1. read the input geometry,
  2. run a wB97X/6-31G(d) SP to create an initial checkpoint,
  3. compute an MLIP Cartesian Hessian at the same geometry,
  4. inject the lower-triangle force constants into an fchk,
  5. unfchk back to a modified checkpoint,
  6. optionally rewrite ts+freq.gjf to a single-link readfc Gaussian job.

The production Gaussian job then has no External route and no nested g16.
"""
from __future__ import annotations

import argparse
import csv
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
CALCALL = HERE / "calcall"
QM_METHOD = "wB97X/6-31G(d)"
INITIAL_STEM = "mlip_readfc_initial"
MODIFIED_STEM = "mlip_readfc"
BOHR_TO_ANG = 0.529177210903


SYMBOL_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Br": 35,
    "I": 53,
}


@dataclass
class Geometry:
    chk_line: str
    nproc_line: str
    charge: int
    mult: int
    atomic_nums: np.ndarray
    coord_lines: list[str]
    coords_ang: np.ndarray


class MlipHessianComputer:
    """Load the MLIP checkpoint once and compute Hessians for many geometries."""

    def __init__(self) -> None:
        self.horm_root = Path(
            os.environ.get("HORM_ROOT", "/home/wuping/GitHub/HORM")
        ).resolve()
        self.device = os.environ.get("HORM_DEVICE", "cpu")
        self.pm = None
        self.model_name = ""
        self.horm = None
        self._old_cwd = os.getcwd()
        self._old_path = list(sys.path)

    def __enter__(self) -> "MlipHessianComputer":
        sys.path.insert(0, str(HERE))
        import horm_external as horm  # noqa: PLC0415

        self.horm = horm
        if self.device == "cuda" and not horm.torch.cuda.is_available():
            print("Warning: CUDA unavailable; falling back to CPU", file=sys.stderr)
            self.device = "cpu"
        if not self.horm_root.is_dir():
            raise RuntimeError(f"HORM_ROOT does not exist: {self.horm_root}")
        os.chdir(self.horm_root)
        if str(self.horm_root) not in sys.path:
            sys.path.insert(0, str(self.horm_root))
        self.pm, self.device, self.model_name = horm._load_horm_model(
            self.horm_root, self.device
        )
        return self

    def __exit__(self, *exc_info) -> None:
        os.chdir(self._old_cwd)
        sys.path[:] = self._old_path

    def hessian_hartree_bohr2(
        self, atomic_nums: np.ndarray, coords_ang: np.ndarray
    ) -> np.ndarray:
        if self.horm is None:
            raise RuntimeError("MLIP model is not loaded")
        coords_bohr = coords_ang / BOHR_TO_ANG
        _, _, hess_ev_ang2 = self.horm._forward_predict(
            self.pm,
            self.model_name,
            self.device,
            atomic_nums,
            coords_bohr,
            need_hessian=True,
        )
        if hess_ev_ang2 is None:
            raise RuntimeError("MLIP did not return a Hessian")
        hess = hess_ev_ang2 * self.horm.HESS_CONV
        return 0.5 * (hess + hess.T)


def run(cmd: list[str], cwd: Path, *, dry_run: bool = False) -> None:
    print(f"[cmd] ({cwd}) {' '.join(cmd)}", file=sys.stderr)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_ts_gjf(path: Path) -> Geometry:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    chk_line = ""
    nproc_line = "%nproc=96"
    charge = mult = None
    charge_idx = None

    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.lower().startswith("%chk=") and not chk_line:
            chk_line = s
        elif s.lower().startswith("%nproc=") and nproc_line == "%nproc=96":
            nproc_line = s
        elif re.match(r"^[+-]?\d+\s+\d+\s*$", s):
            charge_idx = i
            charge, mult = [int(x) for x in s.split()[:2]]
            break

    if not chk_line or charge_idx is None or charge is None or mult is None:
        raise ValueError(f"Cannot parse Gaussian TS input: {path}")

    atomic_nums: list[int] = []
    coord_lines: list[str] = []
    coords: list[list[float]] = []
    for ln in lines[charge_idx + 1 :]:
        if not ln.strip() or ln.strip().startswith("--Link"):
            break
        parts = ln.split()
        if len(parts) < 4:
            break
        atom = parts[0]
        try:
            z = int(float(atom))
        except ValueError:
            sym = atom[:1].upper() + atom[1:].lower()
            if sym not in SYMBOL_TO_Z:
                raise ValueError(f"Unsupported atom symbol {atom!r} in {path}")
            z = SYMBOL_TO_Z[sym]
        xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        atomic_nums.append(z)
        coord_lines.append(ln)
        coords.append(xyz)

    if not atomic_nums:
        raise ValueError(f"No Cartesian coordinates found in {path}")

    return Geometry(
        chk_line=chk_line,
        nproc_line=nproc_line,
        charge=charge,
        mult=mult,
        atomic_nums=np.asarray(atomic_nums, dtype=np.int64),
        coord_lines=coord_lines,
        coords_ang=np.asarray(coords, dtype=np.float64),
    )


def write_sp_gjf(ts_dir: Path, geom: Geometry, nproc: str, method: str) -> Path:
    path = ts_dir / f"{INITIAL_STEM}.gjf"
    coords = "\n".join(geom.coord_lines)
    text = f"""%chk={INITIAL_STEM}.chk
{nproc}
#P {method} SP

QM SP checkpoint for MLIP readfc injection

{geom.charge} {geom.mult}
{coords}

"""
    path.write_text(text, encoding="utf-8")
    return path


def fchk_array_block(header: str, values: Iterable[float]) -> list[str]:
    vals = list(values)
    lines = [f"{header:40}   R   N={len(vals):12d}"]
    for i in range(0, len(vals), 5):
        lines.append("".join(f"{v:16.8E}" for v in vals[i : i + 5]))
    return lines


def read_fchk_array(path: Path, header_prefix: str) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if not line.startswith(header_prefix):
            continue
        m = re.search(r"N=\s*(\d+)", line)
        if not m:
            raise RuntimeError(f"Malformed fchk array header: {line}")
        count = int(m.group(1))
        vals: list[float] = []
        idx = i + 1
        while idx < len(lines) and len(vals) < count:
            vals.extend(float(x) for x in lines[idx].split())
            idx += 1
        if len(vals) < count:
            raise RuntimeError(
                f"{path.name}: {header_prefix!r} expected {count}, got {len(vals)}"
            )
        return np.asarray(vals[:count], dtype=np.float64)
    raise RuntimeError(f"{path.name}: missing fchk array {header_prefix!r}")


def read_fchk_int_array(path: Path, header_prefix: str) -> np.ndarray:
    arr = read_fchk_array(path, header_prefix)
    return np.rint(arr).astype(np.int64)


def force_constants_lower_triangle(hess: np.ndarray) -> list[float]:
    dim = hess.shape[0]
    if hess.shape != (dim, dim):
        raise RuntimeError(f"Hessian is not square: {hess.shape}")
    return [float(hess[i, j]) for i in range(dim) for j in range(i + 1)]


def replace_or_insert_force_constants(
    initial_fchk: Path, modified_fchk: Path, hess_hb2: np.ndarray
) -> None:
    lines = initial_fchk.read_text(encoding="utf-8", errors="replace").splitlines()
    dim = hess_hb2.shape[0]
    expected = dim * (dim + 1) // 2
    fc_lines = fchk_array_block(
        "Cartesian Force Constants", force_constants_lower_triangle(hess_hb2)
    )
    if len(fc_lines) < 2 or f"N={expected:12d}" not in fc_lines[0]:
        raise RuntimeError("Internal error while formatting force constants")

    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("Cartesian Force Constants"):
            start = i
            break

    if start is not None:
        m = re.search(r"N=\s*(\d+)", lines[start])
        if not m:
            raise RuntimeError(f"Malformed force constants header in {initial_fchk}")
        count = int(m.group(1))
        end = start + 1
        seen = 0
        while end < len(lines) and seen < count:
            seen += len(lines[end].split())
            end += 1
        out = lines[:start] + fc_lines + lines[end:]
    else:
        insert_at = len(lines)
        for i, ln in enumerate(lines):
            if ln.startswith("Nonadiabatic coupling") or ln.startswith("Dipole Moment"):
                insert_at = i
                break
        out = lines[:insert_at] + fc_lines + lines[insert_at:]

    modified_fchk.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_readfc_gjf(ts_dir: Path, geom: Geometry, nproc: str, method: str) -> Path:
    path = ts_dir / "ts+freq.gjf"
    text = f"""%chk={MODIFIED_STEM}.chk
{nproc}
#P {method} opt(ts,readfc,noeigen,nomicro,maxcycles=150) freq Geom=AllCheck Guess=TCheck

QM TS opt+freq with MLIP initial Cartesian FC

"""
    path.write_text(text, encoding="utf-8")
    return path


def process_one(
    rxn_dir: Path,
    computer: MlipHessianComputer,
    *,
    nproc: str,
    method: str,
    rewrite_gjf: bool,
    force_sp: bool,
    dry_run: bool,
) -> str:
    ts_dir = rxn_dir / "TS+Freq"
    gjf = ts_dir / "ts+freq.gjf"
    if not gjf.is_file():
        return "missing_gjf"

    try:
        geom = parse_ts_gjf(gjf)
    except ValueError:
        # If this directory has already been rewritten to a Geom=AllCheck
        # readfc job, recover the original geometry from the SP input created
        # by an earlier preprocessing pass.
        sp_source = ts_dir / f"{INITIAL_STEM}.gjf"
        if not sp_source.is_file():
            raise
        geom = parse_ts_gjf(sp_source)
    natoms = len(geom.atomic_nums)
    dim = 3 * natoms

    sp_gjf = write_sp_gjf(ts_dir, geom, nproc, method)
    initial_chk = ts_dir / f"{INITIAL_STEM}.chk"
    initial_fchk = ts_dir / f"{INITIAL_STEM}.fchk"
    modified_fchk = ts_dir / f"{MODIFIED_STEM}.fchk"
    modified_chk = ts_dir / f"{MODIFIED_STEM}.chk"

    if force_sp or not initial_chk.is_file():
        run(["g16", str(sp_gjf.name)], ts_dir, dry_run=dry_run)
    if dry_run:
        return "dry_run"
    if not initial_chk.is_file():
        raise RuntimeError(f"SP did not create {initial_chk}")

    # Version 3 formatted checkpoints are the G09/G16-compatible text format
    # for modern checkpoint contents, including force-constant data.
    run(["formchk", "-3", str(initial_chk.name), str(initial_fchk.name)], ts_dir)
    # Use the geometry stored in the checkpoint, not the original gjf text.
    # formchk may translate/rotate coordinates; readfc requires FC to match
    # the checkpoint geometry and orientation exactly.
    fchk_atomic_nums = read_fchk_int_array(initial_fchk, "Atomic numbers")
    if not np.array_equal(fchk_atomic_nums, geom.atomic_nums):
        raise RuntimeError(
            f"{rxn_dir.name}: atom order changed between gjf and fchk"
        )
    coords_bohr = read_fchk_array(initial_fchk, "Current cartesian coordinates")
    if coords_bohr.size != dim:
        raise RuntimeError(
            f"{rxn_dir.name}: expected {dim} fchk coordinates, got {coords_bohr.size}"
        )
    coords_ang = coords_bohr.reshape(natoms, 3) * BOHR_TO_ANG
    hess = computer.hessian_hartree_bohr2(fchk_atomic_nums, coords_ang)
    if hess.shape != (dim, dim):
        raise RuntimeError(f"{rxn_dir.name}: expected Hessian {(dim, dim)}, got {hess.shape}")
    replace_or_insert_force_constants(initial_fchk, modified_fchk, hess)
    if modified_chk.exists():
        modified_chk.unlink()
    run(["unfchk", str(modified_fchk.name), str(modified_chk.name)], ts_dir)
    if not modified_chk.is_file():
        raise RuntimeError(f"unfchk did not create {modified_chk}")
    if rewrite_gjf:
        write_readfc_gjf(ts_dir, geom, nproc, method)
    return "ok"


def check_ts_freq_log(check_py: Path, log: Path) -> tuple[str, str]:
    if not check_py.is_file():
        return "0", "missing_check_script"
    try:
        cp = subprocess.run(
            [sys.executable, str(check_py), str(log)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception as exc:
        return "0", f"check_failed:{type(exc).__name__}"
    if cp.returncode != 0 and not cp.stdout.strip():
        return "0", "check_failed"
    out = cp.stdout.strip().split("\t", 1)
    if len(out) == 1:
        return out[0], ""
    return out[0], out[1].replace(",", ";")


def run_ts_freq(
    rxn_dir: Path,
    *,
    check_py: Path,
    skip_completed: bool,
) -> tuple[int | None, str, str, int, int]:
    ts_dir = rxn_dir / "TS+Freq"
    gjf = ts_dir / "ts+freq.gjf"
    log = ts_dir / "ts+freq.log"
    stamp = ts_dir / ".ts_freq_strict_ok"
    if not gjf.is_file():
        return None, "0", "missing_gjf", 0, 0
    if skip_completed and stamp.is_file() and log.is_file():
        strict_ok, reason = check_ts_freq_log(check_py, log)
        return None, strict_ok, reason, 0, 1

    t0 = subprocess.run(["date", "+%s"], text=True, stdout=subprocess.PIPE, check=True)
    start = int(t0.stdout.strip())
    with gjf.open("r", encoding="utf-8") as gjf_fh, log.open(
        "w", encoding="utf-8"
    ) as fh:
        cp = subprocess.run(
            ["g16"],
            cwd=str(ts_dir),
            stdin=gjf_fh,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    t1 = subprocess.run(["date", "+%s"], text=True, stdout=subprocess.PIPE, check=True)
    elapsed = int(t1.stdout.strip()) - start
    strict_ok, reason = check_ts_freq_log(check_py, log)
    if strict_ok == "1":
        stamp.touch()
    else:
        stamp.unlink(missing_ok=True)
    return cp.returncode, strict_ok, reason, elapsed, 0


def rxn_dirs(root: Path, selected: list[str], limit: int | None) -> list[Path]:
    if selected:
        dirs = [root / s for s in selected]
    else:
        dirs = sorted(
            [p for p in root.iterdir() if p.is_dir() and p.name.startswith("rxn")],
            key=lambda p: int(p.name[3:]) if p.name[3:].isdigit() else p.name,
        )
    if limit is not None:
        dirs = dirs[:limit]
    return dirs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calcall-root", type=Path, default=CALCALL)
    ap.add_argument("--rxn", action="append", default=[], help="rxn name, repeatable")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--method", default=QM_METHOD)
    ap.add_argument("--nproc", default="%nproc=96")
    ap.add_argument("--rewrite-gjf", action="store_true")
    ap.add_argument("--force-sp", action="store_true", help="rerun the initial QM SP")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--run-ts-freq",
        action="store_true",
        help="after each rxn is prepared, immediately run ts+freq.gjf",
    )
    ap.add_argument(
        "--skip-completed",
        action="store_true",
        help="with --run-ts-freq, skip logs that already have .ts_freq_strict_ok",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record per-rxn preparation errors and continue with the next rxn",
    )
    ap.add_argument(
        "--run-stats-dir",
        type=Path,
        default=HERE / "slurm_logs" / "run_stats",
    )
    ap.add_argument("--job-tag", default=os.environ.get("SLURM_JOB_ID", "manual"))
    args = ap.parse_args()

    root = args.calcall_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"calcall root does not exist: {root}")
    if not shutil.which("g16"):
        raise SystemExit("g16 not found in PATH")
    if not shutil.which("formchk"):
        raise SystemExit("formchk not found in PATH")
    if not shutil.which("unfchk"):
        raise SystemExit("unfchk not found in PATH")

    dirs = rxn_dirs(root, args.rxn, args.limit)
    if not dirs:
        raise SystemExit("no rxn directories selected")

    print(
        f"Preparing {len(dirs)} rxn dirs | method={args.method} | "
        f"rewrite_gjf={args.rewrite_gjf} | run_ts_freq={args.run_ts_freq}",
        file=sys.stderr,
    )

    csv_path = None
    csv_fh = None
    writer = None
    check_py = HERE / "slurm_logs" / "check_ts_freq_log.py"
    if args.run_ts_freq and not args.dry_run:
        args.run_stats_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.run_stats_dir / f"ts_freq_pipeline_{args.job_tag}.csv"
        csv_fh = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            csv_fh,
            fieldnames=[
                "stem",
                "prep_status",
                "g16_exit",
                "strict_ok",
                "reason",
                "elapsed_s",
                "skipped",
            ],
        )
        writer.writeheader()

    total_run = total_skip = total_ok = 0
    manager = contextlib.nullcontext(None) if args.dry_run else MlipHessianComputer()
    try:
        with manager as computer:
            for d in dirs:
                print(f"[{d.name}] start", file=sys.stderr)
                try:
                    status = process_one(
                        d,
                        computer,
                        nproc=args.nproc,
                        method=args.method,
                        rewrite_gjf=args.rewrite_gjf,
                        force_sp=args.force_sp,
                        dry_run=args.dry_run,
                    )
                except Exception as exc:
                    print(f"[{d.name}] error: {exc}", file=sys.stderr)
                    if not args.continue_on_error:
                        raise
                    status = f"prep_error:{type(exc).__name__}"
                    if writer is not None:
                        writer.writerow(
                            {
                                "stem": d.name,
                                "prep_status": status,
                                "g16_exit": "",
                                "strict_ok": "0",
                                "reason": str(exc).replace(",", ";"),
                                "elapsed_s": 0,
                                "skipped": 0,
                            }
                        )
                        csv_fh.flush()
                    continue
                print(f"[{d.name}] prep={status}", file=sys.stderr)

                if args.run_ts_freq and not args.dry_run:
                    g16_exit, strict_ok, reason, elapsed, skipped = run_ts_freq(
                        d,
                        check_py=check_py,
                        skip_completed=args.skip_completed,
                    )
                    total_skip += int(skipped)
                    if not skipped:
                        total_run += 1
                    total_ok += int(strict_ok == "1")
                    print(
                        f"[{d.name}] g16_exit={g16_exit} ok={strict_ok} "
                        f"t={elapsed}s skipped={skipped} reason={reason}",
                        file=sys.stderr,
                    )
                    if writer is not None:
                        writer.writerow(
                            {
                                "stem": d.name,
                                "prep_status": status,
                                "g16_exit": "" if g16_exit is None else g16_exit,
                                "strict_ok": strict_ok,
                                "reason": reason,
                                "elapsed_s": elapsed,
                                "skipped": skipped,
                            }
                        )
                        csv_fh.flush()
    finally:
        if csv_fh is not None:
            csv_fh.close()

    if csv_path is not None:
        summary = {
            "job_tag": args.job_tag,
            "strict_ok": total_ok,
            "total": len(dirs),
            "g16_runs": total_run,
            "skipped": total_skip,
        }
        summary_path = args.run_stats_dir / f"ts_freq_pipeline_{args.job_tag}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"pipeline CSV={csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
