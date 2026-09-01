"""Gaussian input and formatted-checkpoint helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

PERIODIC_SYMBOLS = (
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na",
    "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti",
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
    "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
    "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
SYMBOL_TO_Z = {symbol.lower(): z for z, symbol in enumerate(PERIODIC_SYMBOLS) if symbol}


@dataclass(frozen=True)
class Geometry:
    charge: int
    multiplicity: int
    atomic_numbers: np.ndarray
    coordinates_angstrom: np.ndarray
    coordinate_lines: tuple[str, ...]


def parse_gaussian_input(path: Path) -> Geometry:
    """Read charge, multiplicity, atom order, and Cartesian coordinates."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    charge_index = None
    charge = multiplicity = None
    for index, line in enumerate(lines):
        match = re.fullmatch(r"\s*([+-]?\d+)\s+(\d+)\s*", line)
        if match:
            charge_index = index
            charge, multiplicity = map(int, match.groups())
            break
    if charge_index is None or charge is None or multiplicity is None:
        raise ValueError(f"Cannot find charge/multiplicity in {path}")

    numbers: list[int] = []
    coordinates: list[list[float]] = []
    coordinate_lines: list[str] = []
    for line in lines[charge_index + 1 :]:
        if not line.strip() or line.lstrip().startswith("--Link1--"):
            break
        fields = line.split()
        if len(fields) < 4:
            break
        atom = fields[0]
        try:
            atomic_number = int(float(atom))
        except ValueError:
            symbol = re.sub(r"[^A-Za-z].*$", "", atom).lower()
            if symbol not in SYMBOL_TO_Z:
                raise ValueError(f"Unsupported atom label {atom!r} in {path}")
            atomic_number = SYMBOL_TO_Z[symbol]
        try:
            xyz = [float(value.replace("D", "E")) for value in fields[1:4]]
        except ValueError as exc:
            raise ValueError(f"Invalid Cartesian coordinate line: {line!r}") from exc
        numbers.append(atomic_number)
        coordinates.append(xyz)
        coordinate_lines.append(line)
    if not numbers:
        raise ValueError(f"No Cartesian coordinates found in {path}")
    return Geometry(
        charge=charge,
        multiplicity=multiplicity,
        atomic_numbers=np.asarray(numbers, dtype=np.int64),
        coordinates_angstrom=np.asarray(coordinates, dtype=np.float64),
        coordinate_lines=tuple(coordinate_lines),
    )


def read_fchk_array(path: Path, header_prefix: str) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if not line.startswith(header_prefix):
            continue
        match = re.search(r"N=\s*(\d+)", line)
        if not match:
            raise RuntimeError(f"Malformed fchk array header: {line}")
        count = int(match.group(1))
        values: list[float] = []
        cursor = index + 1
        while cursor < len(lines) and len(values) < count:
            values.extend(
                float(token.replace("D", "E").replace("d", "e"))
                for token in lines[cursor].split()
            )
            cursor += 1
        if len(values) < count:
            raise RuntimeError(
                f"{path}: {header_prefix!r} expected {count}, got {len(values)}"
            )
        return np.asarray(values[:count], dtype=np.float64)
    raise RuntimeError(f"{path}: missing fchk array {header_prefix!r}")


def read_fchk_int_array(path: Path, header_prefix: str) -> np.ndarray:
    return np.rint(read_fchk_array(path, header_prefix)).astype(np.int64)


def lower_triangle(matrix: np.ndarray) -> list[float]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got {matrix.shape}")
    return [float(matrix[i, j]) for i in range(matrix.shape[0]) for j in range(i + 1)]


def _fchk_array_block(header: str, values: Iterable[float]) -> list[str]:
    values = list(values)
    output = [f"{header:40}   R   N={len(values):12d}"]
    for index in range(0, len(values), 5):
        output.append("".join(f"{value:16.8E}" for value in values[index : index + 5]))
    return output


def inject_cartesian_force_constants(
    source_fchk: Path,
    target_fchk: Path,
    hessian_hartree_bohr2: np.ndarray,
) -> None:
    """Replace or insert Gaussian's lower-triangular Cartesian force constants."""
    hessian = np.asarray(hessian_hartree_bohr2, dtype=np.float64)
    hessian = 0.5 * (hessian + hessian.T)
    block = _fchk_array_block("Cartesian Force Constants", lower_triangle(hessian))
    lines = source_fchk.read_text(encoding="utf-8", errors="replace").splitlines()

    start = next(
        (i for i, line in enumerate(lines) if line.startswith("Cartesian Force Constants")),
        None,
    )
    if start is None:
        insert_at = next(
            (
                i
                for i, line in enumerate(lines)
                if line.startswith("Nonadiabatic coupling") or line.startswith("Dipole Moment")
            ),
            len(lines),
        )
        output = lines[:insert_at] + block + lines[insert_at:]
    else:
        match = re.search(r"N=\s*(\d+)", lines[start])
        if not match:
            raise RuntimeError(f"Malformed force-constant header in {source_fchk}")
        expected = int(match.group(1))
        end = start + 1
        observed = 0
        while end < len(lines) and observed < expected:
            observed += len(lines[end].split())
            end += 1
        if observed < expected:
            raise RuntimeError(f"Truncated force-constant block in {source_fchk}")
        output = lines[:start] + block + lines[end:]
    target_fchk.write_text("\n".join(output) + "\n", encoding="utf-8")


def write_sp_input(
    path: Path,
    geometry: Geometry,
    *,
    method: str,
    checkpoint: str = "initial.chk",
    nproc: int = 24,
    memory: str | None = None,
) -> None:
    link0 = [f"%chk={checkpoint}", f"%nprocshared={nproc}"]
    if memory:
        link0.append(f"%mem={memory}")
    coordinates = "\n".join(geometry.coordinate_lines)
    path.write_text(
        "\n".join(link0)
        + f"\n#P {method} SP\n\nFixed-geometry QM checkpoint for ML curvature handoff\n\n"
        + f"{geometry.charge} {geometry.multiplicity}\n{coordinates}\n\n",
        encoding="utf-8",
    )


def write_readfc_input(
    path: Path,
    *,
    method: str,
    checkpoint: str = "readfc.chk",
    nproc: int = 24,
    memory: str | None = None,
    max_cycles: int = 150,
) -> None:
    link0 = [f"%chk={checkpoint}", f"%nprocshared={nproc}"]
    if memory:
        link0.append(f"%mem={memory}")
    route = (
        f"#P {method} Opt=(TS,ReadFC,NoEigenTest,NoMicro,MaxCycles={max_cycles}) "
        "Freq Geom=AllCheck Guess=TCheck"
    )
    path.write_text("\n".join(link0) + f"\n{route}\n\n", encoding="utf-8")


def write_irc_input(
    path: Path,
    *,
    method: str,
    checkpoint: Path,
    nproc: int = 24,
    memory: str | None = None,
    max_points: int = 30,
    step_size: int = 15,
) -> None:
    link0 = [f"%oldchk={checkpoint}", "%chk=irc.chk", f"%nprocshared={nproc}"]
    if memory:
        link0.append(f"%mem={memory}")
    route = (
        f"#P {method} IRC=(RCFC,LQA,MaxPoints={max_points},StepSize={step_size}) "
        "Geom=AllCheck Guess=TCheck"
    )
    path.write_text("\n".join(link0) + f"\n{route}\n\n", encoding="utf-8")
