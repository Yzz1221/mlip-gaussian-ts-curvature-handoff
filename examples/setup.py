#!/usr/bin/env python3
"""Verify the bundled HORM source and assemble the EquiformerV2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HORM = ROOT / "horm"
MODEL = ROOT / "ckpt" / "eqv2.ckpt"
HORM_NETWORK = HORM / "nets" / "equiformer_v2" / "equiformer_v2_oc20.py"
EXPECTED_NETWORK_SHA256 = "62c7e11c4dfa15a74be816462049178c100c09ac77f90a9a9046c5de59f73a70"
EXPECTED_MODEL_SHA256 = "6b5adb66776041a45ab85e5e496c3b1e37f2b102be31247384a8ee69ee56016a"
MODEL_PARTS = [ROOT / "ckpt" / f"eqv2.ckpt.{index:02d}.part" for index in range(14)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_network() -> None:
    if not (HORM / "training_module.py").is_file():
        raise RuntimeError("Bundled HORM source is missing from horm/")
    if sha256(HORM_NETWORK) != EXPECTED_NETWORK_SHA256:
        raise RuntimeError("Bundled EquiformerV2 network differs from the production version")
    print("Bundled EquiformerV2 network: verified")


def install_model(local_source: Path | None) -> None:
    if MODEL.exists() and sha256(MODEL) == EXPECTED_MODEL_SHA256:
        print("EquiformerV2 checkpoint: already verified")
        return
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL.with_suffix(".ckpt.partial")
    digest = hashlib.sha256()
    parts = [local_source] if local_source is not None else MODEL_PARTS
    if local_source is None:
        missing = [path.name for path in parts if not path.is_file()]
        if missing:
            raise RuntimeError(f"The repository is missing checkpoint parts: {missing}")
    with temporary.open("wb") as target:
        for path in parts:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
                    digest.update(chunk)
    if digest.hexdigest() != EXPECTED_MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Checkpoint SHA256 does not match the model used in the study")
    temporary.replace(MODEL)
    print(f"EquiformerV2 checkpoint: verified at {MODEL}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path, help="Use an existing local eqv2.ckpt instead of bundled parts")
    args = parser.parse_args()
    verify_network()
    install_model(args.model_source)


if __name__ == "__main__":
    main()
