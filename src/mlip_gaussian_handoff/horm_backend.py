"""Thin adapter around a local HORM checkout.

The adapter intentionally does not vendor HORM or its checkpoints. It follows
the inference branches used by HORM's evaluation code and returns energy in eV,
forces in eV/angstrom, and an optional Cartesian Hessian in eV/angstrom^2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

MODEL_ALIASES = {
    "eqv2": "eqv2.ckpt",
    "eqv2_orig": "eqv2_orig.ckpt",
    "alpha": "alpha.ckpt",
    "alpha_orig": "alpha_orig.ckpt",
    "left": "left.ckpt",
    "left_orig": "left_orig.ckpt",
    "left-df": "left-df.ckpt",
    "left-df_orig": "left-df_orig.ckpt",
}


def resolve_checkpoint(horm_root: Path, checkpoint: Path | None = None) -> Path:
    raw = checkpoint or (
        Path(os.environ["HORM_CHECKPOINT"]) if os.environ.get("HORM_CHECKPOINT") else None
    )
    if raw is not None:
        candidates = [raw, horm_root / raw, horm_root / "ckpt" / raw]
        for candidate in candidates:
            candidate = candidate.expanduser().resolve()
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"HORM checkpoint not found: {raw}")
    alias = os.environ.get("HORM_MODEL", "eqv2").lower()
    if alias not in MODEL_ALIASES:
        raise ValueError(f"Unknown HORM_MODEL={alias!r}; choose {sorted(MODEL_ALIASES)}")
    candidate = (horm_root / "ckpt" / MODEL_ALIASES[alias]).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"HORM checkpoint not found: {candidate}")
    return candidate


def _model_name(module: object) -> str:
    config = getattr(module, "model_config", None)
    if isinstance(config, dict) and config.get("name"):
        return str(config["name"])
    hparams = module.hparams
    config = hparams.model_config if hasattr(hparams, "model_config") else hparams["model_config"]
    name = config.get("name") if isinstance(config, dict) else getattr(config, "name", None)
    if name:
        return str(name)
    inferred = {
        "EquiformerV2_OC20": "EquiformerV2",
        "AlphaNet": "AlphaNet",
    }.get(type(module.potential).__name__)
    if inferred:
        return inferred
    raise RuntimeError("Cannot determine model family from the HORM checkpoint")


def _leftnet_one_hot(atomic_numbers: np.ndarray) -> np.ndarray:
    encoder = {
        1: [1, 0, 0, 0, 0],
        6: [0, 1, 0, 0, 0],
        7: [0, 0, 1, 0, 0],
        8: [0, 0, 0, 1, 0],
        9: [0, 0, 0, 0, 1],
    }
    try:
        return np.asarray([encoder[int(z)] for z in atomic_numbers], dtype=np.float32)
    except KeyError as exc:
        raise ValueError("LEFTNet checkpoints support only H/C/N/O/F") from exc


class HORMBackend:
    """Load one HORM checkpoint and reuse it for multiple geometries."""

    def __init__(
        self,
        root: Path | None = None,
        checkpoint: Path | None = None,
        device: str | None = None,
    ) -> None:
        root_value = root or (Path(os.environ["HORM_ROOT"]) if os.environ.get("HORM_ROOT") else None)
        if root_value is None:
            raise ValueError("Set HORM_ROOT or pass root=...")
        self.root = Path(root_value).expanduser().resolve()
        self.checkpoint = resolve_checkpoint(self.root, checkpoint)
        self.device = device or os.environ.get("HORM_DEVICE", "cpu")
        self.module = None
        self.name = ""
        self._old_cwd: str | None = None
        self._old_path: list[str] | None = None

    def __enter__(self) -> "HORMBackend":
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {self.device}, but CUDA is unavailable")
        self._old_cwd = os.getcwd()
        self._old_path = list(sys.path)
        os.chdir(self.root)
        sys.path.insert(0, str(self.root))
        from training_module import PotentialModule

        torch.set_float32_matmul_precision("high")
        self.module = PotentialModule.load_from_checkpoint(
            str(self.checkpoint), strict=False, map_location=self.device
        )
        self.module.eval()
        self.module.potential.to(self.device)
        self.name = _model_name(self.module)
        return self

    def __exit__(self, *_: object) -> None:
        if self._old_cwd is not None:
            os.chdir(self._old_cwd)
        if self._old_path is not None:
            sys.path[:] = self._old_path

    def predict(
        self,
        atomic_numbers: np.ndarray,
        coordinates_angstrom: np.ndarray,
        *,
        hessian: bool = False,
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        if self.module is None:
            raise RuntimeError("Use HORMBackend as a context manager")
        import torch
        from torch_geometric.data import Data
        from training_module import remove_mean_batch

        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
        coordinates = np.asarray(coordinates_angstrom, dtype=np.float64)
        natoms = len(atomic_numbers)
        if coordinates.shape != (natoms, 3):
            raise ValueError(f"Expected coordinates {(natoms, 3)}, got {coordinates.shape}")

        pos = torch.tensor(coordinates, dtype=torch.float32, device=self.device)
        batch_index = torch.zeros(natoms, dtype=torch.long, device=self.device)
        natoms_tensor = torch.tensor([natoms], dtype=torch.long, device=self.device)
        atomic_tensor = torch.tensor(atomic_numbers, dtype=torch.long, device=self.device)
        fields = {
            "pos": pos,
            "batch": batch_index,
            "natoms": natoms_tensor,
            "z": atomic_tensor,
        }
        if self.name in ("LEFTNet", "LEFTNet-df"):
            fields.update(
                one_hot=torch.tensor(_leftnet_one_hot(atomic_numbers), device=self.device),
                charges=torch.tensor(atomic_numbers, dtype=torch.float32, device=self.device),
                ae=torch.tensor([natoms], dtype=torch.long, device=self.device),
            )
        else:
            fields["ae"] = torch.zeros(1, dtype=torch.float32, device=self.device)
        data = Data(**fields)
        data.pos.requires_grad_(hessian or self.name in ("LEFTNet", "LEFTNet-df"))
        data.pos = remove_mean_batch(data.pos, data.batch)
        data.pos.requires_grad_(hessian or self.name in ("LEFTNet", "LEFTNet-df"))

        if not hessian and self.name not in ("LEFTNet", "LEFTNet-df"):
            with torch.no_grad():
                energy, forces = self.module.potential.forward(data)
        elif self.name == "LEFTNet":
            energy, forces = self.module.potential.forward_autograd(data)
        else:
            energy, forces = self.module.potential.forward(data)

        energy_ev = float(energy.squeeze().detach().cpu())
        forces_ev_ang = forces.detach().cpu().numpy().astype(np.float64).reshape(natoms, 3)
        hessian_ev_ang2 = None
        if hessian:
            from eval import compute_hessian

            value = compute_hessian(data.pos, energy, forces)
            hessian_ev_ang2 = value.detach().cpu().numpy().astype(np.float64)
            hessian_ev_ang2 = 0.5 * (hessian_ev_ang2 + hessian_ev_ang2.T)
        return energy_ev, forces_ev_ang, hessian_ev_ang2
