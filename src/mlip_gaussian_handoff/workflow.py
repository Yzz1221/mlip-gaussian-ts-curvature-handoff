"""End-to-end preparation of a Gaussian ReadFC transition-state job."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .gaussian_io import (
    inject_cartesian_force_constants,
    parse_gaussian_input,
    read_fchk_array,
    read_fchk_int_array,
    write_readfc_input,
    write_sp_input,
)
from .horm_backend import HORMBackend
from .units import BOHR_TO_ANGSTROM, hessian_to_gaussian


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], cwd: Path, *, stdin: Path | None = None, stdout: Path | None = None) -> float:
    start = time.perf_counter()
    input_handle = stdin.open("r", encoding="utf-8") if stdin else None
    output_handle = stdout.open("w", encoding="utf-8") if stdout else None
    try:
        subprocess.run(
            command,
            cwd=cwd,
            stdin=input_handle,
            stdout=output_handle,
            stderr=subprocess.STDOUT if output_handle else None,
            check=True,
        )
    finally:
        if input_handle:
            input_handle.close()
        if output_handle:
            output_handle.close()
    return time.perf_counter() - start


def prepare_handoff(
    input_gjf: Path,
    output_dir: Path,
    *,
    method: str,
    nproc: int,
    memory: str | None,
    max_cycles: int,
    run_sp: bool,
) -> dict[str, object]:
    """Create a checkpoint containing the one-shot ML Cartesian Hessian."""
    input_gjf = input_gjf.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = parse_gaussian_input(input_gjf)
    initial_gjf = output_dir / "initial.gjf"
    initial_chk = output_dir / "initial.chk"
    initial_fchk = output_dir / "initial.fchk"
    readfc_fchk = output_dir / "readfc.fchk"
    readfc_chk = output_dir / "readfc.chk"
    ts_freq_gjf = output_dir / "ts_freq.gjf"

    write_sp_input(
        initial_gjf,
        geometry,
        method=method,
        checkpoint=initial_chk.name,
        nproc=nproc,
        memory=memory,
    )
    timings: dict[str, float] = {}
    if run_sp:
        for executable in ("g16", "formchk", "unfchk"):
            if not shutil.which(executable):
                raise RuntimeError(f"Required executable not found: {executable}")
        timings["qm_sp_seconds"] = run_command(
            ["g16"], output_dir, stdin=initial_gjf, stdout=output_dir / "initial.log"
        )
    if not initial_chk.is_file():
        raise FileNotFoundError(
            f"{initial_chk} is missing; pass --run-sp or provide the checkpoint"
        )
    timings["formchk_seconds"] = run_command(
        ["formchk", "-3", initial_chk.name, initial_fchk.name], output_dir
    )

    atomic_numbers = read_fchk_int_array(initial_fchk, "Atomic numbers")
    if not np.array_equal(atomic_numbers, geometry.atomic_numbers):
        raise RuntimeError("Atom order changed between the input and checkpoint")
    coordinates_bohr = read_fchk_array(initial_fchk, "Current cartesian coordinates")
    coordinates_angstrom = coordinates_bohr.reshape(len(atomic_numbers), 3) * BOHR_TO_ANGSTROM

    start = time.perf_counter()
    with HORMBackend() as backend:
        _, _, hessian_ev_ang2 = backend.predict(
            atomic_numbers, coordinates_angstrom, hessian=True
        )
        checkpoint = backend.checkpoint
        model_name = backend.name
        device = backend.device
    timings["ml_hessian_seconds"] = time.perf_counter() - start
    if hessian_ev_ang2 is None:
        raise RuntimeError("The ML backend did not return a Hessian")
    hessian_hartree_bohr2 = hessian_to_gaussian(hessian_ev_ang2)
    inject_cartesian_force_constants(initial_fchk, readfc_fchk, hessian_hartree_bohr2)
    if readfc_chk.exists():
        readfc_chk.unlink()
    timings["unfchk_seconds"] = run_command(
        ["unfchk", readfc_fchk.name, readfc_chk.name], output_dir
    )
    write_readfc_input(
        ts_freq_gjf,
        method=method,
        checkpoint=readfc_chk.name,
        nproc=nproc,
        memory=memory,
        max_cycles=max_cycles,
    )
    metadata: dict[str, object] = {
        "workflow": "MLIP-OneShot-ReadFC",
        "input_gjf": str(input_gjf),
        "input_gjf_sha256": sha256(input_gjf),
        "method": method,
        "nproc": nproc,
        "memory": memory,
        "max_cycles": max_cycles,
        "atom_count": len(atomic_numbers),
        "atom_order": atomic_numbers.tolist(),
        "ml_model_family": model_name,
        "ml_checkpoint": str(checkpoint),
        "ml_checkpoint_sha256": sha256(checkpoint),
        "ml_device": device,
        "raw_hessian_unit": "eV/angstrom^2",
        "gaussian_hessian_unit": "hartree/bohr^2",
        "symmetrization": "0.5 * (H + H.T)",
        "timings_seconds": timings,
        "prepared_files": {
            "initial_checkpoint": initial_chk.name,
            "readfc_checkpoint": readfc_chk.name,
            "ts_freq_input": ts_freq_gjf.name,
        },
    }
    (output_dir / "handoff_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
