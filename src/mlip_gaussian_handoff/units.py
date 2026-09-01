"""Unit conversions used by the Gaussian handoff."""

from __future__ import annotations

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
EV_TO_HARTREE = 1.0 / 27.211386245988
FORCE_EV_ANG_TO_GRADIENT_HARTREE_BOHR = EV_TO_HARTREE * BOHR_TO_ANGSTROM
HESSIAN_EV_ANG2_TO_HARTREE_BOHR2 = EV_TO_HARTREE * BOHR_TO_ANGSTROM**2


def symmetrize_hessian(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square Hessian, got {matrix.shape}")
    return 0.5 * (matrix + matrix.T)


def hessian_to_gaussian(matrix_ev_ang2: np.ndarray) -> np.ndarray:
    return symmetrize_hessian(matrix_ev_ang2) * HESSIAN_EV_ANG2_TO_HARTREE_BOHR2


def force_to_gaussian_gradient(forces_ev_ang: np.ndarray) -> np.ndarray:
    return -np.asarray(forces_ev_ang, dtype=np.float64) * FORCE_EV_ANG_TO_GRADIENT_HARTREE_BOHR
