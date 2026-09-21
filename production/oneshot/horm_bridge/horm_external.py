#!/usr/bin/env python3
"""Gaussian 16 Link 402（external）接口：读 Gaussian 写入的 .EIn，写出 .EOu。

================================================================================
一、整体数据流
================================================================================
  主 Gaussian 作业（如 opt(ts) external='./horm.sh'）
       ↓ 每一步优化需要能量/梯度/(可选)Hessian 时
  Gaussian 生成临时目录下的 .EIn → 调用 horm.sh → 本脚本 horm_external.py
       ↓
  本脚本写回 .EOu → Gaussian 读入后继续 Berny 等优化

  因此：「优化器」是主 g16；本脚本只是「被调用的能量/导数提供者」，不负责选步长。

================================================================================
二、当前实现的「混合」策略（与纯 MLIP 全势能不同）
================================================================================
  - 能量、梯度：由【嵌套子进程 g16】做 QM 单点（默认 wB97X/6-31G(d)，
    route=force），从 formchk 得到的 fchk 里读取。即 E/F 来自量子化学，不是 MLIP。
  - 注意：主 Gaussian 作业的 route 里【不要】同时写泛函/基组（如 wB97X/6-31G(d)）与
    External='./horm.sh' —— G16 会在主作业里直接跑内部 SCF（log 见「SCF Done」），
    而不会调用 Link 402 external。要用 horm 时主 #P 只保留 opt(ts,…) 与 external；
    嵌套 QM 级别用环境变量 QM_METHOD 或同目录 TS.gjf（解析）、默认见 horm.sh。
  - Hessian（仅当 Gaussian 请求二阶导，EIn 里 derivs=2，例如 calcfc/calcall）：
    始终由 MLIP（`compute_hessian`）在同一几何上计算；不做嵌套 g16 freq。
  - 嵌套 QM 临时目录：默认在「含 .EIn 的目录」下建 tmp_qm_runs（守护进程下亦如此，避免落在 HORM_ROOT）。
        可用环境变量 HORM_QM_TMP_ROOT 指定父目录（其下仍会创建 tmp_qm_runs）。

  MLIP 部分与 HORM 仓库 eval.py 对齐：
    LEFTNet → forward_autograd；其它 → forward（EquiformerV2、AlphaNet、LEFTNet-df 等）。

================================================================================
三、checkpoint 与单位
================================================================================
  权重选择（优先级）：HORM_CHECKPOINT 指定文件 > HORM_MODEL 别名。
    别名 eqv2 默认使用本文件内固定绝对路径（非 HORM_ROOT/ckpt）；其它别名仍用 HORM_ROOT/ckpt/*.ckpt。

  单位换算（MLIP 内部多为 eV / eV/Å / eV/Å²）：
    EV_TO_HARTREE、BOHR_TO_ANG → FORCE_CONV、HESS_CONV 用于与 Gaussian（Hartree/Bohr）对齐。
  模型输出为力 F = -dE/dx；Gaussian .EOu 要的是梯度 dE/dx = -F（在 _write_eou 前已按 QM 路径处理）。

================================================================================
四、可选：Unix 套接字守护进程（horm.sh 里 HORM_USE_DAEMON=1）
================================================================================
  子进程模式每次调用都重新 import、可能重复加载大模型；守护进程只加载一次 PotentialModule，
  长优化时显著减少启动开销。见 _daemon_loop / --client。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import subprocess
import tempfile
import time
import uuid

import numpy as np
import torch

# ---------- 单位常数（Gaussian 用 Hartree 与 Bohr；HORM 常用 eV 与 Å）----------
BOHR_TO_ANG = 0.529177210903  # 1 Bohr = 0.529177... Å
EV_TO_HARTREE = 1.0 / 27.211386245988  # 1 eV 对应的 Hartree
# 力：eV/Å → Hartree/Bohr（梯度 dE/dx 与模型输出的力差一个负号，在各自路径中处理）
FORCE_CONV = EV_TO_HARTREE * BOHR_TO_ANG
# Hessian：eV/Å² → Hartree/Bohr²
HESS_CONV = EV_TO_HARTREE * (BOHR_TO_ANG**2)

# HORM_MODEL=eqv2 且未设 HORM_CHECKPOINT 时使用的固定权重（不走 HORM_ROOT/ckpt）
EQV2_CKPT_PATH = Path(
    "/home/wuping/WorkBench/Github/ReactBench/ckpt/horm/EFH/eqv2.ckpt"
)

# HORM_MODEL 别名 → HORM_ROOT/ckpt/ 下实际文件名（eqv2 见上固定路径；可自行扩展）
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


def _format_d(x: float) -> str:
    """Gaussian 文本接口习惯用 Fortran 双精度样式（D 指数）。"""
    return f"{x:20.12E}".replace("E", "D")


def _gaussian_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "E"))


def _write_triplets(fh, values: Iterable[float]) -> None:
    """把浮点序列按每行三个数写入（.EOu 中部分块要求三列格式）。"""
    vals = list(values)
    for i in range(0, len(vals), 3):
        fh.write("".join(_format_d(v) for v in vals[i : i + 3]) + "\n")


def _read_fchk_array(
    lines: List[str], header_prefix: str, expected: int | None = None
) -> np.ndarray:
    """从 formchk 文本中按节名（行首）解析 N= 后面的展平浮点数组。"""
    for i, line in enumerate(lines):
        if line.startswith(header_prefix):
            parts = line.split()
            if "N=" not in parts:
                raise RuntimeError(
                    f"fchk 节 {header_prefix!r} 格式错误：{line.rstrip()}"
                )
            count = int(parts[parts.index("N=") + 1])
            vals: List[float] = []
            idx = i + 1
            while len(vals) < count:
                vals.extend(float(x) for x in lines[idx].split())
                idx += 1
            arr = np.asarray(vals[:count], dtype=np.float64)
            if expected is not None and count != expected:
                raise RuntimeError(
                    f"{header_prefix!r} 期望 {expected} 个值，实际 {count} 个"
                )
            return arr
    raise RuntimeError(f"未在 fchk 中找到节：{header_prefix!r}")


def _extract_qm_data_from_log(log_text: str, natoms: int) -> Tuple[float, np.ndarray]:
    """从 Gaussian log 里直接解析最终 SCF 能量和 Forces 块。

    这是嵌套 QM 的快速路径：避免 formchk/fchk 读写。
    Gaussian log 中的 ``Forces (Hartrees/Bohr)`` 是物理力 F = -dE/dx，
    这里为了与 fchk 路径保持一致，返回梯度 dE/dx。
    """
    lines = log_text.splitlines()
    energy_h: float | None = None
    force_blocks: list[np.ndarray] = []

    energy_re = re.compile(
        r"^\s*(?:SCF Done:\s+E\([^)]+\)\s*=\s*|Energy=\s*)([-+0-9.DdEe]+)"
    )
    force_header_re = re.compile(r"^\s*Center\s+Atomic\s+Forces \(Hartrees/Bohr\)\s*$")
    force_row_re = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+([-+0-9.DdEe]+)\s+([-+0-9.DdEe]+)\s+([-+0-9.DdEe]+)\s*$"
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        m = energy_re.match(line)
        if m:
            energy_h = _gaussian_float(m.group(1))
            i += 1
            continue
        if force_header_re.match(line):
            i += 1
            while i < len(lines) and not force_row_re.match(lines[i]):
                i += 1
            rows: list[list[float]] = []
            while i < len(lines):
                rm = force_row_re.match(lines[i])
                if not rm:
                    break
                rows.append(
                    [
                        _gaussian_float(rm.group(3)),
                        _gaussian_float(rm.group(4)),
                        _gaussian_float(rm.group(5)),
                    ]
                )
                i += 1
            if rows:
                force_blocks.append(np.asarray(rows, dtype=np.float64))
            continue
        i += 1

    if energy_h is None:
        raise RuntimeError("未在 Gaussian log 中找到 SCF Done")
    if not force_blocks:
        raise RuntimeError("未在 Gaussian log 中找到 Forces (Hartrees/Bohr) 块")

    forces = force_blocks[-1]
    if forces.shape != (natoms, 3):
        raise RuntimeError(
            f"Gaussian log forces 形状错误：期望 ({natoms}, 3)，实际 {forces.shape}"
        )
    grads_hb = -forces
    return energy_h, grads_hb


def _resolve_qm_method() -> Tuple[str, dict[str, str]]:
    """解析嵌套 QM 单点所用的计算级别。

    优先级：
    1. QM_METHOD：完整 Gaussian 方法字符串，如 ``M06-2X/def2SVP``
    2. QM_FUNCTIONAL + QM_BASIS：拆分配置
    3. 主 GJF route 行里解析到的方法（如 ``wB97X/6-31G(d)``）
    4. 默认值：wB97X / 6-31G(d)
    """
    raw_method = os.environ.get("QM_METHOD", "").strip()
    if raw_method:
        return raw_method, {
            "source": "QM_METHOD",
            "functional": "",
            "basis": "",
            "method": raw_method,
        }

    functional_raw = os.environ.get("QM_FUNCTIONAL", "").strip()
    basis_raw = os.environ.get("QM_BASIS", "").strip()
    if functional_raw or basis_raw:
        if not functional_raw or not basis_raw:
            raise RuntimeError(
                "QM_FUNCTIONAL 和 QM_BASIS 必须同时为非空，"
                "或直接设置 QM_METHOD。"
            )
        return f"{functional_raw}/{basis_raw}", {
            "source": "QM_FUNCTIONAL+QM_BASIS",
            "functional": functional_raw,
            "basis": basis_raw,
            "method": f"{functional_raw}/{basis_raw}",
        }

    gjf_method, gjf_path = _resolve_qm_method_from_gjf()
    if gjf_method:
        functional, basis = gjf_method.split("/", 1)
        return gjf_method, {
            "source": f"GJF_ROUTE({gjf_path})",
            "functional": functional,
            "basis": basis,
            "method": gjf_method,
        }

    functional = "wB97X"
    basis = "6-31G(d)"
    return f"{functional}/{basis}", {
        "source": "default",
        "functional": functional,
        "basis": basis,
        "method": f"{functional}/{basis}",
    }


def _extract_method_from_route(route_text: str) -> str:
    """从 Gaussian route 文本中提取第一个“方法/基组”token。"""
    route = route_text.strip()
    route = re.sub(r"^\s*#\s*[PpNnTt]?\s*", "", route)
    for tok in route.split():
        if "/" not in tok:
            continue
        # 跳过注释或明显非方法 token
        if tok.startswith("!") or tok.startswith("//"):
            continue
        cleaned = tok.strip().strip(",;")
        head = cleaned.split("/", 1)[0]
        # external='./horm.sh' 等：第一个 `/` 前含 `=`，不是 functional/basis
        if "=" in head:
            continue
        if cleaned.lower().startswith("external"):
            continue
        if "/" in cleaned and len(head) > 0:
            return cleaned
    return ""


def _read_gjf_route(path: Path) -> str:
    """读取 gjf route 段（从首个 # 开始到空行结束，支持多行 route）。"""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("#"):
            start = i
            break
    if start is None:
        return ""
    route_lines: List[str] = []
    for ln in lines[start:]:
        if not ln.strip():
            break
        route_lines.append(ln.strip())
    return " ".join(route_lines)


def _resolve_qm_method_from_gjf() -> Tuple[str, str]:
    """尝试从主 GJF route 行获取方法/基组，返回 (method, gjf_path_str)。"""
    env_candidates = [
        os.environ.get("QM_GJF", "").strip(),
        os.environ.get("GJF", "").strip(),
    ]
    candidates: List[Path] = []
    for raw in env_candidates:
        if not raw:
            continue
        p = Path(os.path.expanduser(raw))
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.is_file():
            candidates.append(p)

    ts_default = (Path.cwd() / "TS.gjf").resolve()
    if ts_default.is_file():
        candidates.append(ts_default)

    if not candidates:
        gjfs = sorted(Path.cwd().glob("*.gjf"))
        if len(gjfs) == 1:
            candidates.append(gjfs[0].resolve())

    seen: set[Path] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        route = _read_gjf_route(p)
        if not route:
            continue
        method = _extract_method_from_route(route)
        if method:
            return method, str(p)
    return "", ""


def _read_ein(path: Path):
    """解析 Gaussian 写入的 .EIn（Link 402 约定）。

    首行：原子数 natoms、导数阶 derivs（0=只要能量，1=能量+梯度，2=还要 Hessian）、
          电荷、自旋多重度。
    后续 natoms 行：原子序数 Z 与 xyz（Bohr）。
    """
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
    """写 Gaussian 期望的 .EOu 格式。

    第一行四个数：能量 + 三个占位（常为 0）。
    接着每个原子一行：梯度分量（Hartree/Bohr）。
    derivs==2 时：先写两段占位块，再写下三角 Hessian（与文档中 Link 402 一致）。

    grads_hb：Cartesian 梯度 dE/dx（与 QM 子任务从 fchk 读的 Gradient 一致）。
    """
    natoms = grads_hb.shape[0]
    if grads_hb.shape != (natoms, 3):
        raise RuntimeError(f"梯度形状错误：期望 ({natoms}, 3)，实际 {grads_hb.shape}")
    if derivs == 2:
        dim = 3 * natoms
        if hess_hb2 is None:
            raise RuntimeError("derivs=2 需要 Hessian，但未得到 Hessian 数据")
        if hess_hb2.shape != (dim, dim):
            raise RuntimeError(
                f"Hessian 形状错误：期望 ({dim}, {dim})，实际 {hess_hb2.shape}"
            )

    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            fh.write("".join(_format_d(v) for v in (energy_h, 0.0, 0.0, 0.0)) + "\n")
            for i in range(natoms):
                fh.write("".join(_format_d(v) for v in grads_hb[i]) + "\n")

            if derivs == 2:
                _write_triplets(fh, [0.0] * 6)
                _write_triplets(fh, [0.0] * (9 * natoms))

                tril = []
                for i in range(3 * natoms):
                    for j in range(i + 1):
                        tril.append(float(hess_hb2[i, j]))
                _write_triplets(fh, tril)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


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
    if alias == "eqv2":
        p = EQV2_CKPT_PATH.resolve()
        if p.is_file():
            return p
        msg = f"未找到 eqv2 固定权重：{p}（可设置 HORM_CHECKPOINT 指向其它 .ckpt）"
        if hint:
            msg += f"\n{hint}"
        raise RuntimeError(msg)
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
    if os.environ.get("HORM_VERBOSE", "0") == "1":
        alias = os.environ.get("HORM_MODEL", "").strip()
        ck_env = os.environ.get("HORM_CHECKPOINT", "").strip()
        src = (
            f"HORM_CHECKPOINT={ck_env}"
            if ck_env
            else f"HORM_MODEL={alias or 'eqv2(default)'} → {ckpt.name}"
        )
        print(f"HORM: checkpoint file={ckpt} ({src})", file=sys.stderr)
    map_location = device_str if device_str != "cpu" else "cpu"

    import torch
    from training_module import PotentialModule

    torch_threads = os.environ.get("HORM_TORCH_NUM_THREADS", "").strip()
    if torch_threads:
        torch.set_num_threads(int(torch_threads))
    torch.set_float32_matmul_precision("high")

    pm = PotentialModule.load_from_checkpoint(
        str(ckpt), strict=False, map_location=map_location
    )
    pm.eval()
    pm.potential.to(device_str)
    model_name = _get_model_name(pm)
    return pm, device_str, model_name


def _forward_predict(
    pm,
    model_name: str,
    device_str: str,
    atomic_nums: np.ndarray,
    coords_bohr: np.ndarray,
    need_hessian: bool,
) -> Tuple[float, np.ndarray, np.ndarray | None]:
    """在已加载的 PotentialModule 上前向推理（供 daemon 路径复用，避免重复 load ckpt）。

    返回 (energy_ev, forces_ev_per_ang, hess_ev_per_ang2 | None)。
    forces 为物理力 F = -∇E；若 need_hessian，则调用 eval.compute_hessian 对坐标求二阶导。
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
    """非 daemon：每次调用临时 chdir 到 HORM_ROOT、加载 ckpt、跑 _forward_predict。

    need_hessian=True 时返回 MLIP Hessian；混合模式下 E/F 来自 QM，通常仅采用 Hessian。
    """
    horm_root = Path(os.environ.get("HORM_ROOT", "/home/wuping/GitHub/HORM")).resolve()
    device_str = os.environ.get("HORM_DEVICE", "cpu")

    if device_str == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device_str = 'cpu'

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


def _log_backend_choice(derivs: int) -> None:
    if os.environ.get("HORM_VERBOSE", "0") != "1":
        return
    parts = [
        f"external mode: derivs={derivs}",
        "energy=Gaussian",
        "gradient=Gaussian",
        (
            "hessian=not requested"
            if derivs != 2
            else "hessian=MLIP"
        ),
    ]
    print(" | ".join(parts), file=sys.stderr)


def _log_qm_route(method_info: dict[str, str], route: str) -> None:
    """记录当前嵌套 QM 子任务的 route 与来源，便于排查。"""
    if os.environ.get("HORM_VERBOSE", "0") != "1":
        return

    parts = [
        f"nested QM route=#p {method_info['method']} {route}",
        f"source={method_info['source']}",
    ]
    if method_info.get("functional"):
        parts.append(f"functional={method_info['functional']}")
    if method_info.get("basis"):
        parts.append(f"basis={method_info['basis']}")
    print(" | ".join(parts), file=sys.stderr)


def _build_hybrid_outputs(
    atomic_nums: np.ndarray,
    coords_bohr: np.ndarray,
    charge: int,
    spin: int,
    derivs: int,
    *,
    ml_predictor,
    qm_scratch_parent: Path | None = None,
) -> Tuple[float, np.ndarray, np.ndarray | None, dict[str, float]]:
    """混合势能核心：E 与 grad 来自子进程 QM(force)；derivs=2 时 Hessian 仅由 MLIP 计算。

    derivs!=2：只跑 _run_qm_energy_force，Hessian 不返回（None）。
    derivs==2：QM force + MLIP Hessian（对称化），能量仍为 QM 单点能量。
    ml_predictor：可注入 _predict_horm 或 daemon 内的 lambda。
    qm_scratch_parent：嵌套 g16 临时目录的基准路径（一般为当前步 .EIn 的父目录）。
    """
    metrics: dict[str, float] = {"qm_time_s": 0.0, "mlip_time_s": 0.0}
    t0 = time.perf_counter()
    if derivs != 2:
        qm_energy_h, qm_grads_hb, reuse_metrics = _run_qm_energy_force(
            atomic_nums, coords_bohr, charge, spin, scratch_parent=qm_scratch_parent
        )
        metrics["qm_time_s"] = time.perf_counter() - t0
        metrics.update(reuse_metrics)
        _log_backend_choice(derivs)
        return qm_energy_h, qm_grads_hb, None, metrics

    qm_t0 = time.perf_counter()
    qm_energy_h, qm_grads_hb, reuse_metrics = _run_qm_energy_force(
        atomic_nums,
        coords_bohr,
        charge,
        spin,
        scratch_parent=qm_scratch_parent,
    )
    metrics["qm_time_s"] = time.perf_counter() - qm_t0
    metrics.update(reuse_metrics)
    ml_t0 = time.perf_counter()
    _, _, ml_hess_ev_ang2 = ml_predictor(
        atomic_nums, coords_bohr, need_hessian=True
    )
    metrics["mlip_time_s"] = time.perf_counter() - ml_t0
    if ml_hess_ev_ang2 is None:
        raise RuntimeError("MLIP Hessian 路径未返回 Hessian 数据")
    hess_hb2 = ml_hess_ev_ang2 * HESS_CONV
    hess_hb2 = 0.5 * (hess_hb2 + hess_hb2.T)  # 数值对称化
    _log_backend_choice(derivs)
    return qm_energy_h, qm_grads_hb, hess_hb2, metrics


def _append_external_stats(
    ein: Path,
    eou: Path,
    *,
    natoms: int,
    derivs: int,
    charge: int,
    spin: int,
    total_time_s: float,
    metrics: dict[str, float],
    ok: bool,
    error: str = "",
) -> None:
    """Append one JSONL timing record per external call.

    Set HORM_STATS_JSONL=0 to disable. If unset, horm.sh sets it to the EIn
    directory's horm_external_stats.jsonl.
    """
    raw = os.environ.get("HORM_STATS_JSONL", "").strip()
    if raw == "0":
        return
    path = Path(raw).expanduser() if raw else ein.parent / "horm_external_stats.jsonl"
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        "ein": str(ein),
        "eou": str(eou),
        "natoms": natoms,
        "derivs": derivs,
        "charge": charge,
        "spin": spin,
        "qm_time_s": round(float(metrics.get("qm_time_s", 0.0)), 6),
        "mlip_time_s": round(float(metrics.get("mlip_time_s", 0.0)), 6),
        "qm_reuse_enabled": int(metrics.get("qm_reuse_enabled", 0.0)),
        "qm_reuse_attempted": int(metrics.get("qm_reuse_attempted", 0.0)),
        "qm_reuse_used": int(metrics.get("qm_reuse_used", 0.0)),
        "qm_reuse_fallback": int(metrics.get("qm_reuse_fallback", 0.0)),
        "qm_scf_attempts": int(metrics.get("qm_scf_attempts", 0.0)),
        "qm_scf_xqc_used": int(metrics.get("qm_scf_xqc_used", 0.0)),
        "total_time_s": round(float(total_time_s), 6),
        "horm_qm_nproc": os.environ.get("HORM_QM_NPROC", ""),
        "horm_device": os.environ.get("HORM_DEVICE", ""),
        "ok": ok,
        "error": error,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ein_to_eou_pipeline(ein: Path, eou: Path) -> None:
    """单次非 daemon 调用的主流程：EIn → _build_hybrid_outputs → EOu。"""
    t0 = time.perf_counter()
    natoms, derivs, charge, spin, atomic_nums, coords_bohr = _read_ein(ein)
    if natoms != len(atomic_nums):
        raise RuntimeError("内部错误：原子数不一致。")
    qm_energy_h, qm_grads_hb, hess_hb2, metrics = _build_hybrid_outputs(
        atomic_nums,
        coords_bohr,
        charge,
        spin,
        derivs,
        ml_predictor=_predict_horm,
        qm_scratch_parent=ein.parent.resolve(),
    )
    _write_eou(eou, qm_energy_h, qm_grads_hb, hess_hb2, derivs)
    _append_external_stats(
        ein,
        eou,
        natoms=natoms,
        derivs=derivs,
        charge=charge,
        spin=spin,
        total_time_s=time.perf_counter() - t0,
        metrics=metrics,
        ok=True,
    )

def _daemon_loop(sock_path: str) -> None:
    """常驻进程：监听 Unix 套接字；每请求读两行（EIn 路径、EOu 路径）。

    启动时加载一次模型；循环里用已加载的 pm 调 _forward_predict，避免每步重新 load ckpt。
    成功发 OK\\n，异常发 ERR\\n + JSON（供 horm.sh --client 侧解析）。
    """
    horm_root = Path(os.environ.get("HORM_ROOT", "/home/wuping/GitHub/HORM")).resolve()
    device_str = os.environ.get("HORM_DEVICE", "cpu")
    if not horm_root.is_dir():
        raise RuntimeError(f"HORM_ROOT 不存在：{horm_root}")

    old_cwd = os.getcwd()
    old_path = list(sys.path)
    srv: socket.socket | None = None
    try:
        os.chdir(horm_root)
        if str(horm_root) not in sys.path:
            sys.path.insert(0, str(horm_root))

        if os.path.exists(sock_path):
            os.unlink(sock_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(sock_path)
        os.chmod(sock_path, 0o600)
        srv.listen(8)

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

                t0 = time.perf_counter()
                natoms, derivs, charge, spin, atomic_nums, coords_bohr = _read_ein(ein)
                qm_energy_h, qm_grads_hb, hess_hb2, metrics = _build_hybrid_outputs(
                    atomic_nums,
                    coords_bohr,
                    charge,
                    spin,
                    derivs,
                    ml_predictor=lambda nums, coords, need_hessian: _forward_predict(
                        pm, model_name, device_str, nums, coords, need_hessian
                    ),
                    qm_scratch_parent=ein.parent.resolve(),
                )
                _write_eou(eou, qm_energy_h, qm_grads_hb, hess_hb2, derivs)
                _append_external_stats(
                    ein,
                    eou,
                    natoms=natoms,
                    derivs=derivs,
                    charge=charge,
                    spin=spin,
                    total_time_s=time.perf_counter() - t0,
                    metrics=metrics,
                    ok=True,
                )
                conn.sendall(b"OK\n")
            except Exception as exc:
                err_obj = {"error": str(exc), "type": type(exc).__name__}
                conn.sendall(b"ERR\n" + json.dumps(err_obj).encode("utf-8") + b"\n")
            finally:
                conn.close()
    finally:
        if srv is not None:
            srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        os.chdir(old_cwd)
        sys.path[:] = old_path

def _client_request(sock_path: str, ein: Path, eou: Path) -> None:
    """向 _daemon_loop 发一次计算请求（与 horm.sh HORM_USE_DAEMON=1 配合）。"""
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

def _extract_qm_data_from_fchk(
    fchk_path: Path, natoms: int
) -> Tuple[float, np.ndarray]:
    """从子任务 formchk 结果中取 Total Energy、Cartesian Gradient。"""
    lines = fchk_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    energy_h: float | None = None
    for line in lines:
        if line.startswith("Total Energy"):
            energy_h = float(line.split()[-1])
            break
    if energy_h is None:
        raise RuntimeError(f"未在 fchk 中找到 Total Energy：{fchk_path}")

    grads_flat = _read_fchk_array(lines, "Cartesian Gradient", expected=3 * natoms)
    grads_hb = grads_flat.reshape(-1, 3)

    return energy_h, grads_hb


def _nested_qm_tmp_base_dir(scratch_parent: Path | None) -> Path:
    """嵌套 g16 临时目录（在其下再建 uuid 子目录）。

    优先级：
    1. 环境变量 ``HORM_QM_TMP_ROOT``：解析为该路径下的 ``tmp_qm_runs``（便于统一大盘路径）。
    2. ``scratch_parent``：通常为当前步 ``.EIn`` 所在目录（daemon 下与单次调用行为一致）。
    3. ``Path.cwd()``：与旧版兼容。
    """
    env_raw = os.environ.get("HORM_QM_TMP_ROOT", "").strip()
    if env_raw:
        root = Path(env_raw).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        base_dir = root / "tmp_qm_runs"
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir
    base = scratch_parent if scratch_parent is not None else Path.cwd()
    base_dir = (base / "tmp_qm_runs").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _qm_reuse_signature(
    *,
    atomic_nums: np.ndarray,
    charge: int,
    spin: int,
    qm_method: str,
    route: str,
) -> dict[str, object]:
    """Signature for deciding whether a previous nested-QM checkpoint is reusable."""
    return {
        "atomic_nums": [int(z) for z in atomic_nums.tolist()],
        "charge": int(charge),
        "spin": int(spin),
        "qm_method": qm_method,
        "route": route,
    }


def _qm_reuse_paths(base_dir: Path) -> tuple[Path, Path]:
    return base_dir / "last_qm.chk", base_dir / "last_qm_signature.json"


def _signature_matches(sig_path: Path, signature: dict[str, object]) -> bool:
    if not sig_path.is_file():
        return False
    try:
        old = json.loads(sig_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return old == signature


def _write_nested_qm_gjf(
    *,
    gjf_path: Path,
    chk_name: str,
    oldchk_name: str | None,
    nproc: str,
    qm_method: str,
    route: str,
    scf_keyword: str,
    charge: int,
    spin: int,
    atomic_nums: np.ndarray,
    coords_ang: np.ndarray,
) -> None:
    route_bits = route
    if scf_keyword:
        route_bits += f" {scf_keyword}"
    if os.environ.get("HORM_QM_NOSYMM", "0") == "1":
        route_bits += " nosymm"
    if oldchk_name is not None:
        route_bits += " guess=read"
    gjf_content = ""
    if oldchk_name is not None:
        gjf_content += f"%oldchk={oldchk_name}\n"
    gjf_content += f"""%chk={chk_name}
%nproc={nproc}
#p {qm_method} SP {route_bits}

Temp QM calculation

{charge} {spin}
"""
    for z, (x, y, zc) in zip(atomic_nums, coords_ang):
        gjf_content += f"{z}  {x:12.8f}  {y:12.8f}  {zc:12.8f}\n"
    gjf_content += "\n"
    gjf_path.write_text(gjf_content, encoding="utf-8")


def _run_nested_qm_once(
    *,
    workdir: Path,
    gjf_path: Path,
    log_path: Path,
    chk_path: Path,
    fchk_path: Path,
    natoms: int,
) -> Tuple[float, np.ndarray]:
    parse_log_first = os.environ.get("HORM_QM_PARSE_LOG", "1") != "0"
    g16_res = subprocess.run(
        ["g16", gjf_path.name],
        check=True,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_text = g16_res.stdout or ""
    if not log_path.exists() or not log_text:
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        else:
            log_path.write_text(log_text, encoding="utf-8", errors="ignore")
    else:
        log_path.write_text(log_text, encoding="utf-8", errors="ignore")

    if parse_log_first:
        try:
            return _extract_qm_data_from_log(log_text, natoms)
        except Exception:
            pass

    subprocess.run(
        ["formchk", chk_path.name, fchk_path.name],
        check=True,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not fchk_path.exists():
        raise RuntimeError(f"未生成预期的 fchk 文件：{fchk_path}")
    return _extract_qm_data_from_fchk(fchk_path, natoms)


def _nested_qm_scf_keywords() -> list[tuple[str, str]]:
    """Return ordered SCF attempts for nested Gaussian force jobs.

    Default is fast ordinary SCF first, XQC only as fallback. Set
    HORM_QM_SCF_FIRST / HORM_QM_SCF_FALLBACK to override, or set
    HORM_QM_SCF_FALLBACK=0 to disable the retry.
    """
    first = os.environ.get("HORM_QM_SCF_FIRST", "scf=(maxcycle=64)").strip()
    fallback = os.environ.get(
        "HORM_QM_SCF_FALLBACK", "scf=(xqc,maxcycle=128)"
    ).strip()
    attempts = [("regular", first)]
    if fallback and fallback != "0":
        attempts.append(("xqc", fallback))
    return attempts


def _run_qm_job(
    atomic_nums: np.ndarray,
    coords_bohr: np.ndarray,
    charge: int,
    spin: int,
    route: str,
    *,
    scratch_parent: Path | None = None,
) -> Tuple[float, np.ndarray, dict[str, float]]:
    """写 gjf → 调 g16，优先从 log 解析 E/F，失败才 formchk 兜底。

    route：Gaussian 关键词片段，本模块嵌套 QM 仅用于 force 单点（梯度）。
    默认算完删除临时目录；调试可设 HORM_KEEP_QM_TMP=1 保留。
    scratch_parent：嵌套 QM 的 ``tmp_qm_runs`` 父目录（一般为 .EIn 所在目录）。
    """
    qm_method, method_info = _resolve_qm_method()
    nproc = os.environ.get("HORM_QM_NPROC", os.environ.get("OMP_NUM_THREADS", "1"))
    keep_tmp = os.environ.get("HORM_KEEP_QM_TMP", "0") == "1"
    fixed_workdir = os.environ.get("HORM_QM_FIXED_WORKDIR", "0") == "1"

    coords_ang = coords_bohr * BOHR_TO_ANG
    job_id = uuid.uuid4().hex[:8]
    base_dir = _nested_qm_tmp_base_dir(scratch_parent)
    if fixed_workdir:
        workdir = base_dir / "active_qm"
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = Path(tempfile.mkdtemp(prefix=f"temp_qm_{job_id}_", dir=base_dir))

    stem = f"temp_qm_{job_id}"
    gjf_path = workdir / f"{stem}.gjf"
    log_path = workdir / f"{stem}.log"
    chk_path = workdir / f"{job_id}.chk"
    fchk_path = workdir / f"{stem}.fchk"
    oldchk_path = workdir / "old_qm.chk"
    reuse_enabled = os.environ.get("HORM_QM_REUSE_CHK", "0") == "1"
    reuse_chk, reuse_sig = _qm_reuse_paths(base_dir)
    signature = _qm_reuse_signature(
        atomic_nums=atomic_nums,
        charge=charge,
        spin=spin,
        qm_method=qm_method,
        route=route,
    )
    can_reuse = (
        reuse_enabled
        and reuse_chk.is_file()
        and _signature_matches(reuse_sig, signature)
    )
    reuse_metrics = {
        "qm_reuse_enabled": float(reuse_enabled),
        "qm_reuse_attempted": 0.0,
        "qm_reuse_used": 0.0,
        "qm_reuse_fallback": 0.0,
        "qm_scf_attempts": 0.0,
        "qm_scf_xqc_used": 0.0,
    }
    scf_attempts = _nested_qm_scf_keywords()

    _log_qm_route(method_info, route)

    try:
        if fixed_workdir:
            for child in list(workdir.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        attempts = [True, False] if can_reuse else [False]
        last_reuse_error = ""
        for attempt_reuse in attempts:
            if attempt_reuse:
                reuse_metrics["qm_reuse_attempted"] = 1.0
            if attempt_reuse:
                shutil.copy2(reuse_chk, oldchk_path)
            elif oldchk_path.exists():
                oldchk_path.unlink()
            for scf_label, scf_keyword in scf_attempts:
                reuse_metrics["qm_scf_attempts"] += 1.0
                if scf_label == "xqc":
                    reuse_metrics["qm_scf_xqc_used"] = 1.0
                if gjf_path.exists():
                    gjf_path.unlink()
                if log_path.exists():
                    log_path.unlink()
                if chk_path.exists():
                    chk_path.unlink()
                if fchk_path.exists():
                    fchk_path.unlink()
                _write_nested_qm_gjf(
                    gjf_path=gjf_path,
                    chk_name=chk_path.name,
                    oldchk_name=oldchk_path.name if attempt_reuse else None,
                    nproc=nproc,
                    qm_method=qm_method,
                    route=route,
                    scf_keyword=scf_keyword,
                    charge=charge,
                    spin=spin,
                    atomic_nums=atomic_nums,
                    coords_ang=coords_ang,
                )
                try:
                    result = _run_nested_qm_once(
                        workdir=workdir,
                        gjf_path=gjf_path,
                        log_path=log_path,
                        chk_path=chk_path,
                        fchk_path=fchk_path,
                        natoms=len(atomic_nums),
                    )
                    if reuse_enabled:
                        shutil.copy2(chk_path, reuse_chk)
                        reuse_sig.write_text(
                            json.dumps(signature, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    if attempt_reuse and os.environ.get("HORM_VERBOSE", "0") == "1":
                        print(
                            "nested QM reused checkpoint with guess=read",
                            file=sys.stderr,
                        )
                    if attempt_reuse:
                        reuse_metrics["qm_reuse_used"] = 1.0
                    return result[0], result[1], reuse_metrics
                except subprocess.CalledProcessError as e:
                    last_reuse_error = e.stdout or str(e)
                    if scf_label != "xqc" and len(scf_attempts) > 1:
                        if os.environ.get("HORM_VERBOSE", "0") == "1":
                            print(
                                "nested QM regular SCF failed; retrying with XQC",
                                file=sys.stderr,
                            )
                        continue
                    if not attempt_reuse:
                        raise
                    break
                except Exception as e:
                    last_reuse_error = str(e)
                    if scf_label != "xqc" and len(scf_attempts) > 1:
                        if os.environ.get("HORM_VERBOSE", "0") == "1":
                            print(
                                "nested QM regular SCF path failed; retrying with XQC",
                                file=sys.stderr,
                            )
                        continue
                    if not attempt_reuse:
                        raise
                    break
            if attempt_reuse:
                reuse_metrics["qm_reuse_fallback"] = 1.0
                if os.environ.get("HORM_VERBOSE", "0") == "1":
                    print(
                        "nested QM checkpoint reuse failed; retrying cold start",
                        file=sys.stderr,
                    )
        raise RuntimeError(f"nested QM failed after checkpoint retry: {last_reuse_error}")
    except subprocess.CalledProcessError as e:
        detail = e.stdout or ""
        if log_path.exists():
            detail += "\n--- temp log tail ---\n" + "".join(
                log_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)[-80:]
            )
        raise RuntimeError(
            "临时 QM 计算失败。\n"
            f"workdir: {workdir}\n"
            f"gjf: {gjf_path}\n"
            f"log: {log_path}\n"
            f"chk: {chk_path}\n"
            f"stdout/log:\n{detail}"
        )
    except Exception as e:
        raise RuntimeError(
            "临时 QM 计算流程失败。\n"
            f"workdir: {workdir}\n"
            f"gjf: {gjf_path}\n"
            f"log: {log_path}\n"
            f"chk: {chk_path}\n"
            f"fchk: {fchk_path}\n"
            f"原因: {e}"
        ) from e
    finally:
        if not keep_tmp and not fixed_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def _run_qm_energy_force(
    atomic_nums: np.ndarray,
    coords_bohr: np.ndarray,
    charge: int,
    spin: int,
    *,
    scratch_parent: Path | None = None,
) -> Tuple[float, np.ndarray, dict[str, float]]:
    """嵌套 QM：#p ... force → 只要能量与梯度（混合路径每步都会调用）。"""
    energy_h, grads_hb, reuse_metrics = _run_qm_job(
        atomic_nums,
        coords_bohr,
        charge,
        spin,
        route="force",
        scratch_parent=scratch_parent,
    )
    return energy_h, grads_hb, reuse_metrics


def main() -> int:
    """入口：--daemon 常驻；--client 把 EIn/EOu 发给 daemon；否则直接 _ein_to_eou_pipeline。"""
    p = argparse.ArgumentParser(description="Gaussian external：混合 QM(E/F) + MLIP Hessian")
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
