#!/usr/bin/env python3
"""Run one manuscript workflow from an initial TS-guess XYZ file."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlip_gaussian_handoff.audit import classify_opt_freq  # noqa: E402
from mlip_gaussian_handoff.gaussian_io import (  # noqa: E402
    SYMBOL_TO_Z,
    read_fchk_array,
    read_fchk_int_array,
)
from mlip_gaussian_handoff.units import BOHR_TO_ANGSTROM  # noqa: E402

TEMPLATES = ROOT / "examples" / "gaussian"
HORM = ROOT / "horm"
MODEL = ROOT / "models" / "eqv2.ckpt"
EXPECTED_MODEL_SHA256 = "6b5adb66776041a45ab85e5e496c3b1e37f2b102be31247384a8ee69ee56016a"
EXPECTED_NETWORK_SHA256 = "62c7e11c4dfa15a74be816462049178c100c09ac77f90a9a9046c5de59f73a70"
WORKFLOWS = {
    "calcfc": "gaussian_calcfc",
    "calcall": "gaussian_calcall",
    "oneshot": "mlip_oneshot_readfc",
    "external_calcall": "external_calcall",
}


def read_xyz(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid XYZ file: {path}")
    natoms = int(lines[0].strip())
    atom_lines = [line.split() for line in lines[2:] if line.strip()]
    if len(atom_lines) != natoms:
        raise ValueError(f"{path}: expected {natoms} atoms, found {len(atom_lines)}")
    symbols: list[str] = []
    numbers: list[int] = []
    coordinates: list[list[float]] = []
    for row in atom_lines:
        if len(row) < 4 or row[0].lower() not in SYMBOL_TO_Z:
            raise ValueError(f"Invalid atom record: {row}")
        xyz = [float(value) for value in row[1:4]]
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError("XYZ contains a nonfinite coordinate")
        symbols.append(row[0])
        numbers.append(SYMBOL_TO_Z[row[0].lower()])
        coordinates.append(xyz)
    return np.asarray(numbers, dtype=np.int64), np.asarray(coordinates), tuple(symbols)


def render_input(
    source: Path,
    target: Path,
    cores: int,
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    symbols: tuple[str, ...],
    charge: int,
    multiplicity: int,
) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    lines = [
        f"%nprocshared={cores}" if re.fullmatch(r"%nprocshared=\d+", line) else line
        for line in lines
    ]
    geometry_line = next(
        (index for index, line in enumerate(lines) if re.fullmatch(r"[+-]?\d+\s+\d+", line)),
        None,
    )
    if geometry_line is not None:
        coordinate_lines = [
            f"{symbol:<2s} {xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f}"
            for symbol, xyz in zip(symbols, coordinates)
        ]
        lines = lines[:geometry_line] + [f"{charge} {multiplicity}"] + coordinate_lines
    target.write_text("\n".join(lines) + "\n\n", encoding="utf-8")


def run_command(command: list[str], directory: Path, log: Path | None = None, env=None) -> None:
    output = log.open("w", encoding="utf-8") if log else subprocess.DEVNULL
    try:
        result = subprocess.run(command, cwd=directory, stdout=output,
                                stderr=subprocess.STDOUT if log else None,
                                env=env, check=False)
    finally:
        if log:
            output.close()
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited with {result.returncode}; see {log}")


def run_gaussian(name: str, directory: Path, env=None) -> Path:
    source = directory / f"{name}.gjf"
    log = directory / f"{name}.log"
    with source.open("r", encoding="utf-8") as input_file, log.open("w", encoding="utf-8") as output:
        result = subprocess.run(["g16"], cwd=directory, stdin=input_file, stdout=output,
                                stderr=subprocess.STDOUT, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"Gaussian exited with {result.returncode}; see {log}")
    return log


def kabsch_row(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Same proper-rotation alignment as the four-core production runner."""
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


def prepare_oneshot(
    directory: Path, atomic_numbers: np.ndarray, coordinates: np.ndarray, cores: int
) -> float:
    for executable in ("g16", "formchk", "unfchk"):
        if not shutil.which(executable):
            raise RuntimeError(f"{executable} is not available on PATH")
    if not HORM.is_dir() or not MODEL.is_file():
        raise RuntimeError("Run python examples/setup.py to prepare HORM and eqv2.ckpt")
    if not production_network_installed():
        raise RuntimeError("Run python examples/setup.py to install the production EquiformerV2 network")

    start = time.perf_counter()
    run_gaussian("initial_sp", directory)
    run_command(["formchk", "-3", "mlip_readfc_initial.chk", "mlip_readfc_initial.fchk"], directory)

    bridge = ROOT / "production" / "oneshot" / "horm_bridge"
    sys.path.insert(0, str(bridge))
    from prepare_mlip_readfc import (  # noqa: PLC0415
        MlipHessianComputer, replace_or_insert_force_constants,
    )

    env_before = {key: os.environ.get(key) for key in ("HORM_ROOT", "HORM_CHECKPOINT", "HORM_DEVICE")}
    os.environ["HORM_ROOT"] = str(HORM)
    os.environ["HORM_CHECKPOINT"] = str(MODEL)
    os.environ["HORM_DEVICE"] = "cpu"
    try:
        with MlipHessianComputer() as computer:
            computer.horm.torch.set_num_threads(cores)
            _, _, raw = computer.horm._forward_predict(
                computer.pm, computer.model_name, computer.device,
                atomic_numbers, coordinates / BOHR_TO_ANGSTROM, need_hessian=True,
            )
            if raw is None:
                raise RuntimeError("EquiformerV2 did not return a Hessian")
            if hasattr(raw, "detach"):
                raw = raw.detach().cpu().numpy()
            hessian = np.asarray(raw, dtype=np.float64) * computer.horm.HESS_CONV
            hessian = 0.5 * (hessian + hessian.T)
    finally:
        for key, value in env_before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    fchk = directory / "mlip_readfc_initial.fchk"
    checkpoint_numbers = read_fchk_int_array(fchk, "Atomic numbers")
    checkpoint_coordinates = read_fchk_array(fchk, "Current cartesian coordinates").reshape(-1, 3)
    checkpoint_coordinates *= BOHR_TO_ANGSTROM
    if not np.array_equal(checkpoint_numbers, atomic_numbers):
        raise RuntimeError("Atom ordering changed between the XYZ and Gaussian checkpoint")
    rotation, alignment_rmsd = kabsch_row(coordinates, checkpoint_coordinates)
    if alignment_rmsd > 1.0e-5:
        raise RuntimeError(f"Gaussian checkpoint alignment RMSD {alignment_rmsd:.3e} Å")
    transform = np.kron(np.eye(len(atomic_numbers)), rotation.T)
    rotated = transform @ hessian @ transform.T
    rotated = 0.5 * (rotated + rotated.T)
    replace_or_insert_force_constants(fchk, directory / "mlip_readfc.fchk", rotated)
    run_command(["unfchk", "mlip_readfc.fchk", "mlip_readfc.chk"], directory)
    print(f"OneShot checkpoint prepared; alignment RMSD {alignment_rmsd:.3e} Å")
    return time.perf_counter() - start


def start_external_daemon(directory: Path, env: dict[str, str]) -> tuple[subprocess.Popen, object]:
    socket_name = f"/tmp/mlip-horm-{os.getpid()}-{hashlib.sha1(str(directory).encode()).hexdigest()[:8]}.sock"
    env["HORM_DAEMON_SOCKET"] = socket_name
    env["HORM_USE_DAEMON"] = "1"
    log_handle = (directory / "horm_daemon.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(directory / "horm_external.py"), "--daemon", "--socket", socket_name],
        cwd=HORM, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(f"HORM daemon exited; see {directory / 'horm_daemon.log'}")
        if Path(socket_name).exists():
            try:
                with socket.socket(socket.AF_UNIX) as connection:
                    connection.settimeout(1)
                    connection.connect(socket_name)
                return process, log_handle
            except OSError:
                pass
        time.sleep(0.2)
    process.terminate()
    log_handle.close()
    raise TimeoutError(f"HORM daemon did not become ready; see {directory / 'horm_daemon.log'}")


def production_network_installed() -> bool:
    installed = HORM / "nets" / "equiformer_v2" / "equiformer_v2_oc20.py"
    if not installed.is_file():
        return False
    return hashlib.sha256(installed.read_bytes()).hexdigest() == EXPECTED_NETWORK_SHA256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, choices=WORKFLOWS)
    parser.add_argument("--xyz", required=True, type=Path, help="Initial TS guess, e.g. data/react_ot/rxn9.xyz")
    parser.add_argument("--charge", required=True, type=int)
    parser.add_argument("--multiplicity", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--external-device", choices=("cpu", "cuda"), default="cuda",
                        help="EquiformerV2 device for External--CalcAll (paper: cuda/A100)")
    parser.add_argument("--dry-run", action="store_true", help="Write inputs without starting Gaussian or HORM")
    parser.add_argument("--skip-irc", action="store_true", help="Stop after Opt+Freq")
    args = parser.parse_args()
    if args.cores < 1 or args.multiplicity < 1:
        parser.error("--cores and --multiplicity must be positive")
    atomic_numbers, coordinates, symbols = read_xyz(args.xyz)
    if not args.dry_run:
        if not shutil.which("g16"):
            raise RuntimeError("Gaussian 16 (g16) is not available on PATH")
        if args.workflow == "oneshot":
            for executable in ("formchk", "unfchk"):
                if not shutil.which(executable):
                    raise RuntimeError(f"{executable} is not available on PATH")
        if args.workflow in ("oneshot", "external_calcall"):
            if not HORM.is_dir() or not MODEL.is_file() or not production_network_installed():
                raise RuntimeError("Run python examples/setup.py before the MLIP workflows")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    template_dir = TEMPLATES / WORKFLOWS[args.workflow]
    for name in ("opt_freq", "irc", "initial_sp"):
        template = template_dir / f"{name}.gjf"
        if template.is_file():
            render_input(template, directory / template.name, args.cores,
                         atomic_numbers, coordinates, symbols, args.charge, args.multiplicity)

    if args.dry_run:
        print(f"Wrote {args.workflow} Gaussian inputs to {directory}")
        return

    daemon = None
    daemon_log = None
    env = os.environ.copy()
    try:
        if args.workflow == "oneshot":
            prepare_oneshot(directory, atomic_numbers, coordinates, args.cores)
        elif args.workflow == "external_calcall":
            if not HORM.is_dir() or not MODEL.is_file():
                raise RuntimeError("Run python examples/setup.py to prepare HORM and eqv2.ckpt")
            if not production_network_installed():
                raise RuntimeError("Run python examples/setup.py to install the production EquiformerV2 network")
            for name in ("horm.sh", "horm_external.py"):
                shutil.copy2(ROOT / "production" / "external" / name, directory / name)
            env.update({
                "HORM_ROOT": str(HORM), "HORM_CHECKPOINT": str(MODEL),
                "HORM_DEVICE": args.external_device, "HORM_PYTHON_BIN": sys.executable,
                "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": str(args.cores),
                "HORM_TORCH_NUM_THREADS": str(args.cores),
            })
            daemon, daemon_log = start_external_daemon(directory, env)

        log = run_gaussian("opt_freq", directory, env=env)
        result = classify_opt_freq(log.read_text(encoding="utf-8", errors="replace"))
        (directory / "opt_freq_status.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Opt+Freq: {result['paper_reason']}")
        if args.skip_irc or result["paper_ok_minus_10"] != 1:
            return
        irc_log = run_gaussian("irc", directory, env=env)
        print(f"IRC completed; inspect {irc_log} and classify both endpoint connectivities")
    finally:
        if daemon is not None:
            daemon.terminate()
            try:
                daemon.wait(timeout=10)
            except subprocess.TimeoutExpired:
                daemon.kill()
            if daemon_log is not None:
                daemon_log.close()
            Path(env["HORM_DAEMON_SOCKET"]).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
