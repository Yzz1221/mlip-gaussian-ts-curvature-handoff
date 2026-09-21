#!/usr/bin/env python3
"""Gaussian external：读取 .EIn，调用 HORM 预训练势，写出 .EOu。

与 HORM 仓库内 eval.py 的约定对齐：
  - LEFTNet     → potential.forward_autograd
  - 其它        → potential.forward（含 EquiformerV2、AlphaNet、LEFTNet-df）

checkpoint 选择（优先级从高到低）：
  1. 环境变量 HORM_CHECKPOINT：.ckpt 的绝对路径，或相对 HORM_ROOT 的路径
  2. 否则 HORM_MODEL：在 HORM_ROOT/ckpt/ 下按别名解析（默认 eqv2 → eqv2.ckpt）

能量与 Gaussian 约定：
  - 能量：eV → Hartree（EV_TO_HARTREE）
  - 一阶导：模型输出为受力 F = -dE/dx（eV/Å），Gaussian .EOu 需梯度 dE/dx，
    故写入前对力取负号再换算为 Hartree/Bohr。
  - 二阶导：Hessian 对称化后按下三角写入。

可选守护进程（见 horm.sh 中 HORM_USE_DAEMON）：模型只加载一次，减少优化多步时的重复启动开销。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

BOHR_TO_ANG = 0.529177210903
EV_TO_HARTREE = 1.0 / 27.211386245988
FORCE_CONV = EV_TO_HARTREE * BOHR_TO_ANG
HESS_CONV = EV_TO_HARTREE * (BOHR_TO_ANG**2)

# 与 /home/wuping/GitHub/HORM/ckpt 目录中文件名对应（可扩展）
MODEL_ALIASES: dict[str, str] = {
    "alpha": "alpha.ckpt",
    "alpha_orig": "alpha_orig.ckpt",
    "eqv2": "eqv2.ckpt",
    "eqv2_orig": "eqv2_orig.ckpt",
    "left": "left.ckpt",
    "left_orig": "left_orig.ckpt",
    "left-df": "left-df.ckpt",
    "left_df": "left-df.ckpt",
    "left-df_orig": "left-df_orig.ckpt",
}


def _startup_log(message: str) -> None:
    if os.environ.get("HORM_STARTUP_LOG", "0") == "1" or os.environ.get("HORM_VERBOSE", "0") == "1":
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}", file=sys.stderr, flush=True)


def _format_d(x: float) -> str:
    return f"{x:20.12E}".replace("E", "D")


def _write_triplets(fh, values: Iterable[float]) -> None:
    vals = list(values)
    for i in range(0, len(vals), 3):
        fh.write("".join(_format_d(v) for v in vals[i : i + 3]) + "\n")


def _read_ein(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    if not lines:
        raise RuntimeError(f"EIn 文件为空：{path}")

    header = lines[0].split()
    if len(header) < 4:
        raise RuntimeError(f"EIn 文件首行格式错误：{path} -> {lines[0].rstrip()}")

    natoms = int(header[0])
    derivs = int(header[1])
    charge = int(float(header[2]))
    spin = int(float(header[3]))

    atom_lines = lines[1 : 1 + natoms]
    if len(atom_lines) != natoms:
        raise RuntimeError(
            f"EIn 原子行数不匹配：期望 {natoms} 行，实际 {len(atom_lines)} 行"
        )

    z_list: List[int] = []
    coords_bohr = np.zeros((natoms, 3), dtype=np.float64)
    for i, line in enumerate(atom_lines):
        parts = line.split()
        if len(parts) < 4:
            raise RuntimeError(f"EIn 第 {i+2} 行原子数据格式错误：{line.rstrip()}")
        z = int(float(parts[0]))
        x, y, zc = float(parts[1]), float(parts[2]), float(parts[3])
        z_list.append(z)
        coords_bohr[i] = [x, y, zc]

    return natoms, derivs, charge, spin, np.array(z_list, dtype=np.int64), coords_bohr


def _leftnet_one_hot(atomic_nums: np.ndarray) -> np.ndarray:
    """LEFTNet / LEFTNet-df 使用的 5 维 one-hot（H, C, N, O, F）。"""
    encoder = {
        1: [1, 0, 0, 0, 0],
        6: [0, 1, 0, 0, 0],
        7: [0, 0, 1, 0, 0],
        8: [0, 0, 0, 1, 0],
        9: [0, 0, 0, 0, 1],
    }
    rows = []
    for z in atomic_nums:
        zi = int(z)
        if zi not in encoder:
            raise ValueError(
                f"LEFTNet 系模型仅支持 H/C/N/O/F（当前 Z={zi}）。"
                "请换用 EquiformerV2 / AlphaNet 的 checkpoint。"
            )
        rows.append(encoder[zi])
    return np.asarray(rows, dtype=np.float32)


def _write_eou(
    path: Path,
    energy_h: float,
    grads_hb: np.ndarray,
    hess_hb2: np.ndarray | None,
    derivs: int,
) -> None:
    """grads_hb：Cartesian 梯度 dE/dx，单位 Hartree/Bohr（与 Gaussian 约定一致）。"""
    natoms = grads_hb.shape[0]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("".join(_format_d(v) for v in (energy_h, 0.0, 0.0, 0.0)) + "\n")
        for i in range(natoms):
            fh.write("".join(_format_d(v) for v in grads_hb[i]) + "\n")

        if derivs == 2:
            _write_triplets(fh, [0.0] * 6)
            _write_triplets(fh, [0.0] * (9 * natoms))

            if hess_hb2 is None:
                raise RuntimeError("derivs=2 需要 Hessian，但未得到 Hessian 数据")

            tril = []
            for i in range(3 * natoms):
                for j in range(i + 1):
                    tril.append(float(hess_hb2[i, j]))
            _write_triplets(fh, tril)


def _list_ckpt_dir(ckpt_dir: Path) -> str:
    if not ckpt_dir.is_dir():
        return ""
    files = sorted(ckpt_dir.glob("*.ckpt"))
    if not files:
        return ""
    return "  目录内现有 .ckpt：" + ", ".join(f.name for f in files)


def _resolve_checkpoint(horm_root: Path) -> Path:
    """解析最终 .ckpt 路径。"""
    ckpt_dir = horm_root / "ckpt"
    hint = _list_ckpt_dir(ckpt_dir)

    raw = os.environ.get("HORM_CHECKPOINT", "").strip()
    if raw:
        p = Path(os.path.expanduser(raw))
        if not p.is_absolute():
            cand = horm_root / raw
            if cand.is_file():
                p = cand.resolve()
            elif (ckpt_dir / raw).is_file():
                p = (ckpt_dir / raw).resolve()
            else:
                p = Path.cwd() / raw
                p = p.resolve()
        else:
            p = p.resolve()
        if p.is_file():
            return p
        msg = f"HORM_CHECKPOINT 指向的文件不存在：{p}"
        if hint:
            msg += f"\n{hint}"
        raise RuntimeError(msg)

    alias = os.environ.get("HORM_MODEL", "eqv2").strip().lower()
    fname = MODEL_ALIASES.get(alias)
    if fname is None:
        known = ", ".join(sorted(MODEL_ALIASES.keys()))
        raise RuntimeError(
            f"未知 HORM_MODEL={alias!r}。已知别名：{known}\n"
            f"或设置 HORM_CHECKPOINT 为具体 .ckpt 路径。\n{hint}"
        )
    p = (ckpt_dir / fname).resolve()
    if p.is_file():
        return p
    msg = f"未找到 {p}（HORM_MODEL={alias!r}）"
    if hint:
        msg += f"\n{hint}"
    raise RuntimeError(msg)


def _get_model_name(pm) -> str:
    """Lightning 保存的 hparams.model_config 可能是展开后的 Equiformer 参数字典（无 'name'），
    但 PotentialModule.model_config 仍保留训练时的 name；优先用后者。"""
    mc = getattr(pm, "model_config", None)
    if isinstance(mc, dict) and mc.get("name"):
        return str(mc["name"])

    h = pm.hparams
    mc2 = h.model_config if hasattr(h, "model_config") else h["model_config"]
    if isinstance(mc2, dict):
        name = mc2.get("name")
    else:
        name = getattr(mc2, "name", None)
    if name:
        return str(name)

    # 最后按 potential 类型推断（与 training_module 分支一致）
    cls = type(pm.potential).__name__
    inferred = {
        "EquiformerV2_OC20": "EquiformerV2",
        "AlphaNet": "AlphaNet",
    }.get(cls)
    if inferred:
        return inferred

    raise RuntimeError(
        "无法从 checkpoint 解析模型名：hparams.model_config 无 name，且 potential 类型 "
        f"{cls!r} 无映射。请检查 HORM ckpt 与 training_module。"
    )


def _load_horm_model(
    horm_root: Path, device_str: str
) -> Tuple[object, str, str]:
    """加载 PotentialModule，返回 (pm, device_str, model_name)。"""
    ckpt = _resolve_checkpoint(horm_root)
    map_location = device_str if device_str != "cpu" else "cpu"

    _startup_log(f"loading HORM model checkpoint={ckpt} device={device_str}")
    _startup_log("import torch")
    import torch
    _startup_log("import PotentialModule")
    from training_module import PotentialModule

    torch.set_float32_matmul_precision("high")

    _startup_log("PotentialModule.load_from_checkpoint start")
    pm = PotentialModule.load_from_checkpoint(
        str(ckpt), strict=False, map_location=map_location
    )
    _startup_log("PotentialModule.load_from_checkpoint done")
    pm.eval()
    _startup_log(f"move potential to {device_str}")
    pm.potential.to(device_str)
    model_name = _get_model_name(pm)
    _startup_log(f"HORM model ready model_name={model_name}")
    return pm, device_str, model_name


def _forward_predict(
    pm,
    model_name: str,
    device_str: str,
    atomic_nums: np.ndarray,
    coords_bohr: np.ndarray,
    need_hessian: bool,
) -> Tuple[float, np.ndarray, np.ndarray | None]:
    """返回 (energy_ev, forces_ev_per_ang, hess_ev_per_ang2 或 None)。

    forces 为物理力 F = -∇E（与 HORM forward 输出一致）；Hessian 在 need_hessian 时计算。
    """
    import torch
    from torch_geometric.data import Data

    from training_module import remove_mean_batch

    coords_ang = coords_bohr * BOHR_TO_ANG
    natoms = int(len(atomic_nums))

    pos = torch.tensor(coords_ang, dtype=torch.float32, device=device_str)
    batch_idx = torch.zeros(natoms, dtype=torch.long, device=device_str)
    natoms_t = torch.tensor([natoms], dtype=torch.long, device=device_str)
    z_t = torch.tensor(atomic_nums, dtype=torch.long, device=device_str)

    if model_name in ("LEFTNet", "LEFTNet-df"):
        oh = torch.tensor(
            _leftnet_one_hot(atomic_nums), dtype=torch.float32, device=device_str
        )
        ch = torch.tensor(atomic_nums, dtype=torch.float32, device=device_str)
        ae = torch.tensor([natoms], dtype=torch.long, device=device_str)
        batch = Data(
            pos=pos,
            batch=batch_idx,
            natoms=natoms_t,
            z=z_t,
            one_hot=oh,
            charges=ch,
            ae=ae,
        )
    else:
        # EquiformerV2_OC20.forward 返回 energy.reshape(data.ae.shape)；OC20 数据里 ae 为每图能量，
        # Gaussian external 无标签，用与图批大小一致的占位形状即可（数值不参与计算）。
        ae_placeholder = torch.zeros(
            natoms_t.shape[0], dtype=torch.float32, device=device_str
        )
        batch = Data(
            pos=pos,
            batch=batch_idx,
            natoms=natoms_t,
            z=z_t,
            ae=ae_placeholder,
        )

    # EquiformerV2 / AlphaNet 等：仅能量+力且不求 Hessian 时用 no_grad，省显存并加速。
    # LEFTNet / LEFTNet-df 保持原 forward 路径（与用户约定一致）。
    use_no_grad = (not need_hessian) and model_name not in ("LEFTNet", "LEFTNet-df")

    if use_no_grad:
        batch.pos = remove_mean_batch(batch.pos, batch.batch)
        batch.z = torch.tensor(atomic_nums, dtype=torch.long, device=device_str)
        with torch.no_grad():
            # 此处 model_name 已排除 LEFTNet / LEFTNet-df，仅 forward()
            ener, force = pm.potential.forward(batch)
    else:
        batch.pos.requires_grad_(True)
        batch.pos = remove_mean_batch(batch.pos, batch.batch)
        batch.pos.requires_grad_(True)
        batch.z = torch.tensor(atomic_nums, dtype=torch.long, device=device_str)

        if model_name == "LEFTNet":
            ener, force = pm.potential.forward_autograd(batch)
        else:
            ener, force = pm.potential.forward(batch)

    energy_ev = float(ener.squeeze().detach().cpu().item())
    forces_ev_ang = force.detach().cpu().numpy().astype(np.float64).reshape(-1, 3)

    hess_ev_ang2 = None
    if need_hessian:
        from eval import compute_hessian

        hess = compute_hessian(batch.pos, ener, force)
        hess_ev_ang2 = hess.detach().cpu().numpy().astype(np.float64)

    return energy_ev, forces_ev_ang, hess_ev_ang2


def _predict_horm(
    atomic_nums: np.ndarray, coords_bohr: np.ndarray, need_hessian: bool
) -> Tuple[float, np.ndarray, np.ndarray | None]:
    horm_root = Path(os.environ.get("HORM_ROOT", "/home/wuping/GitHub/HORM")).resolve()
    device_str = os.environ.get("HORM_DEVICE", "cpu")

    if not horm_root.is_dir():
        raise RuntimeError(f"HORM_ROOT 不存在：{horm_root}")

    old_cwd = os.getcwd()
    old_path = list(sys.path)
    try:
        os.chdir(horm_root)
        if str(horm_root) not in sys.path:
            sys.path.insert(0, str(horm_root))

        pm, device_str, model_name = _load_horm_model(horm_root, device_str)
        return _forward_predict(
            pm, model_name, device_str, atomic_nums, coords_bohr, need_hessian
        )
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def _ein_to_eou_pipeline(ein: Path, eou: Path) -> None:
    """读取 EIn，预测，写出 EOu（含单位换算与 Hessian 对称化）。"""
    natoms, derivs, _charge, _spin, atomic_nums, coords_bohr = _read_ein(ein)
    if natoms != len(atomic_nums):
        raise RuntimeError("内部错误：原子数不一致。")

    need_hessian = derivs == 2
    energy_ev, forces_ev_ang, hess_ev_ang2 = _predict_horm(
        atomic_nums, coords_bohr, need_hessian
    )

    energy_h = energy_ev * EV_TO_HARTREE
    # Gaussian 要梯度 ∇E；模型给的是力 F = -∇E
    grads_hb = -1.0 * forces_ev_ang * FORCE_CONV

    hess_hb2 = None if hess_ev_ang2 is None else (hess_ev_ang2 * HESS_CONV)
    if hess_hb2 is not None:
        hess_hb2 = 0.5 * (hess_hb2 + hess_hb2.T)

    _write_eou(eou, energy_h, grads_hb, hess_hb2, derivs)


def _daemon_loop(sock_path: str) -> None:
    """Unix 域套接字：每行 ein 路径、第二行 eou 路径；成功回复 OK\\n，失败 ERR\\n + JSON。"""
    horm_root = Path(os.environ.get("HORM_ROOT", "/home/wuping/GitHub/HORM")).resolve()
    device_str = os.environ.get("HORM_DEVICE", "cpu")
    if not horm_root.is_dir():
        raise RuntimeError(f"HORM_ROOT 不存在：{horm_root}")

    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o600)
    srv.listen(8)
    _startup_log(f"daemon socket listening: {sock_path}")

    old_cwd = os.getcwd()
    old_path = list(sys.path)
    try:
        os.chdir(horm_root)
        if str(horm_root) not in sys.path:
            sys.path.insert(0, str(horm_root))

        pm, device_str, model_name = _load_horm_model(horm_root, device_str)

        while True:
            conn, _ = srv.accept()
            try:
                data = conn.recv(65536)
                if not data:
                    continue
                text = data.decode("utf-8", errors="replace").strip()
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                if len(lines) < 2:
                    raise RuntimeError("daemon 请求需要两行：EIn 路径与 EOu 路径")
                ein = Path(lines[0])
                eou = Path(lines[1])

                natoms, derivs, _c, _s, atomic_nums, coords_bohr = _read_ein(ein)
                need_hessian = derivs == 2
                energy_ev, forces_ev_ang, hess_ev_ang2 = _forward_predict(
                    pm, model_name, device_str, atomic_nums, coords_bohr, need_hessian
                )

                energy_h = energy_ev * EV_TO_HARTREE
                grads_hb = -1.0 * forces_ev_ang * FORCE_CONV
                hess_hb2 = None if hess_ev_ang2 is None else (hess_ev_ang2 * HESS_CONV)
                if hess_hb2 is not None:
                    hess_hb2 = 0.5 * (hess_hb2 + hess_hb2.T)

                _write_eou(eou, energy_h, grads_hb, hess_hb2, derivs)
                conn.sendall(b"OK\n")
            except Exception as exc:
                err_obj = {"error": str(exc), "type": type(exc).__name__}
                conn.sendall(b"ERR\n" + json.dumps(err_obj).encode("utf-8") + b"\n")
            finally:
                conn.close()
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def _client_request(sock_path: str, ein: Path, eou: Path) -> None:
    payload = f"{ein.resolve()}\n{eou.resolve()}\n".encode("utf-8")
    cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    cli.connect(sock_path)
    cli.sendall(payload)
    resp = cli.recv(65536)
    cli.close()
    if resp.startswith(b"ERR"):
        lines = resp.decode("utf-8", errors="replace").splitlines()
        detail = lines[1] if len(lines) > 1 else resp.decode("utf-8", errors="replace")
        raise RuntimeError(f"HORM daemon 错误：{detail}")


def main() -> int:
    p = argparse.ArgumentParser(description="Gaussian external：HORM 势")
    p.add_argument("ein", nargs="?", type=Path, help="Gaussian .EIn 路径")
    p.add_argument("eou", nargs="?", type=Path, help="待写入的 .EOu 路径")
    p.add_argument(
        "--daemon",
        action="store_true",
        help="启动守护进程（需配合 horm.sh 中 HORM_USE_DAEMON=1）",
    )
    p.add_argument(
        "--socket",
        default=os.environ.get("HORM_DAEMON_SOCKET", "/tmp/horm_g16.sock"),
        help="Unix 域套接字路径",
    )
    p.add_argument(
        "--client",
        action="store_true",
        help="以客户端模式向守护进程发送一次计算请求",
    )
    args = p.parse_args()

    if args.daemon:
        _daemon_loop(args.socket)
        return 0

    if args.client:
        if args.ein is None or args.eou is None:
            print("用法：horm_external.py --client [--socket PATH] <EIn> <EOu>", file=sys.stderr)
            return 2
        _client_request(args.socket, args.ein, args.eou)
        return 0

    if args.ein is None or args.eou is None:
        print("用法：horm_external.py <EIn> <EOu>", file=sys.stderr)
        return 2

    _ein_to_eou_pipeline(args.ein, args.eou)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
