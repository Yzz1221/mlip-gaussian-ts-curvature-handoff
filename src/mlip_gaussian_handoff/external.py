"""Gaussian External-interface adapter for continuous HORM/MLIP controls."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from .horm_backend import HORMBackend
from .units import (
    BOHR_TO_ANGSTROM,
    EV_TO_HARTREE,
    force_to_gaussian_gradient,
    hessian_to_gaussian,
)


def read_ein(path: Path) -> tuple[int, int, int, int, np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise RuntimeError(f"Empty Gaussian EIn file: {path}")
    header = lines[0].split()
    if len(header) < 4:
        raise RuntimeError(f"Malformed Gaussian EIn header: {lines[0]!r}")
    natoms, derivatives, charge, spin = [int(float(value)) for value in header[:4]]
    rows = lines[1 : natoms + 1]
    if len(rows) != natoms:
        raise RuntimeError(f"Expected {natoms} atoms, found {len(rows)}")
    atomic_numbers = np.asarray([int(float(row.split()[0])) for row in rows], dtype=np.int64)
    coordinates_bohr = np.asarray(
        [[float(value.replace("D", "E")) for value in row.split()[1:4]] for row in rows],
        dtype=np.float64,
    )
    return natoms, derivatives, charge, spin, atomic_numbers, coordinates_bohr


def _format_d(value: float) -> str:
    return f"{value:20.12E}".replace("E", "D")


def _write_triplets(handle: object, values: Iterable[float]) -> None:
    values = list(values)
    for index in range(0, len(values), 3):
        handle.write("".join(_format_d(value) for value in values[index : index + 3]) + "\n")


def write_eou(
    path: Path,
    energy_hartree: float,
    gradient_hartree_bohr: np.ndarray,
    hessian_hartree_bohr2: np.ndarray | None,
    derivatives: int,
) -> None:
    natoms = gradient_hartree_bohr.shape[0]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("".join(_format_d(value) for value in (energy_hartree, 0.0, 0.0, 0.0)) + "\n")
        for row in gradient_hartree_bohr:
            handle.write("".join(_format_d(value) for value in row) + "\n")
        if derivatives == 2:
            if hessian_hartree_bohr2 is None:
                raise RuntimeError("Gaussian requested second derivatives but no Hessian was returned")
            _write_triplets(handle, [0.0] * 6)
            _write_triplets(handle, [0.0] * (9 * natoms))
            triangle = [
                float(hessian_hartree_bohr2[i, j])
                for i in range(3 * natoms)
                for j in range(i + 1)
            ]
            _write_triplets(handle, triangle)


def evaluate(ein: Path, eou: Path) -> None:
    natoms, derivatives, _, _, atomic_numbers, coordinates_bohr = read_ein(ein)
    with HORMBackend() as backend:
        energy, forces, hessian = backend.predict(
            atomic_numbers,
            coordinates_bohr.reshape(natoms, 3) * BOHR_TO_ANGSTROM,
            hessian=derivatives == 2,
        )
    write_eou(
        eou,
        energy * EV_TO_HARTREE,
        force_to_gaussian_gradient(forces),
        None if hessian is None else hessian_to_gaussian(hessian),
        derivatives,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ein", type=Path)
    parser.add_argument("eou", type=Path)
    args = parser.parse_args()
    evaluate(args.ein, args.eou)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
