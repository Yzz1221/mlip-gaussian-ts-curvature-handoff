#!/usr/bin/env python3
"""Prepare the pinned HORM source and the released EquiformerV2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HORM = ROOT / "third_party" / "HORM"
MODEL = ROOT / "models" / "eqv2.ckpt"
NETWORK = ROOT / "model_architecture" / "nets" / "equiformer_v2" / "equiformer_v2_oc20.py"
HORM_NETWORK = HORM / "nets" / "equiformer_v2" / "equiformer_v2_oc20.py"
EXPECTED_MODEL_SHA256 = "6b5adb66776041a45ab85e5e496c3b1e37f2b102be31247384a8ee69ee56016a"
RELEASE_API = "https://api.github.com/repos/Yzz1221/mlip-gaussian-ts-curvature-handoff/releases/tags/eqv2-model-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_network() -> None:
    if not (HORM / "training_module.py").is_file():
        raise RuntimeError("HORM submodule is missing: run git submodule update --init --recursive")
    if sha256(HORM_NETWORK) == sha256(NETWORK):
        print("EquiformerV2 network: already matches the production version")
        return
    shutil.copyfile(NETWORK, HORM_NETWORK)
    if sha256(HORM_NETWORK) != sha256(NETWORK):
        raise RuntimeError("EquiformerV2 network installation failed checksum verification")
    print("EquiformerV2 network: installed the documented local adaptation")


def release_assets() -> dict[str, str]:
    request = urllib.request.Request(
        RELEASE_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "mlip-gaussian-setup"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    return {asset["name"]: asset["browser_download_url"] for asset in release["assets"]}


def install_model(local_source: Path | None) -> None:
    if MODEL.exists() and sha256(MODEL) == EXPECTED_MODEL_SHA256:
        print("EquiformerV2 checkpoint: already verified")
        return
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL.with_suffix(".ckpt.partial")
    digest = hashlib.sha256()
    if local_source is not None:
        with local_source.open("rb") as source, temporary.open("wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
                digest.update(chunk)
    else:
        assets = release_assets()
        names = [f"eqv2.ckpt.{index:02d}.part" for index in range(14)]
        missing = [name for name in names if name not in assets]
        if missing:
            raise RuntimeError(f"Release is missing model parts: {missing}")
        with temporary.open("wb") as target:
            for index, name in enumerate(names, start=1):
                request = urllib.request.Request(
                    assets[name], headers={"User-Agent": "mlip-gaussian-setup"}
                )
                with urllib.request.urlopen(request, timeout=120) as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                        digest.update(chunk)
                print(f"Downloaded model part {index}/14", flush=True)
    if digest.hexdigest() != EXPECTED_MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Checkpoint SHA256 does not match the model used in the study")
    temporary.replace(MODEL)
    print(f"EquiformerV2 checkpoint: verified at {MODEL}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", type=Path, help="Use a local eqv2.ckpt instead of downloading")
    args = parser.parse_args()
    install_network()
    install_model(args.model_source)


if __name__ == "__main__":
    main()
