#!/bin/bash

# Gaussian external：HORM 预训练模型（/home/wuping/GitHub/HORM/ckpt 内各 .ckpt）
# Gaussian 16 常见调用：
#   ./horm.sh <EIn> <EOu>
#   ./horm.sh <op> <EIn> <EOu>（op 如 R）
# 亦可能附带更多参数；始终取第 2、3 个为 EIn、EOu（与 G16 Link 402 一致）。
# calcfc 会请求 derivs=2（Hessian），由 horm_external.py 计算。
#
# 性能：HORM_USE_DAEMON=1 时启动 Unix 套接字守护进程，模型只加载一次，适合长优化/多步。
# 默认单次子进程模式（HORM_USE_DAEMON=0）。详见 horm_external.py。

set -euo pipefail

# 与 env_horm.sh 一致：禁止混入 ~/.local，避免 numpy/scipy/torchmetrics 与 Hessian 环境 ABI 冲突
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 解析 .EIn / .EOu 路径
if [ "$#" -eq 2 ]; then
  EIN_FILE="$1"
  EOU_FILE="$2"
elif [ "$#" -ge 3 ]; then
  EIN_FILE="$2"
  EOU_FILE="$3"
  # $4 及以后（EMs/EFC/EUF 等）忽略，由 Gaussian 传入
else
  echo "错误：external 参数不足，至少需要 2 个，实际 $#：$*" >&2
  exit 1
fi

if [ -z "${EIN_FILE}" ] || [ -z "${EOU_FILE}" ]; then
  echo "错误：EIn 或 EOu 路径为空" >&2
  exit 1
fi

ERR_LOG="$(dirname "${EIN_FILE}")/$(basename "${EIN_FILE}" .EIn).horm_stderr.txt"

: "${HORM_ROOT:=/home/wuping/GitHub/HORM}"
: "${HORM_CHECKPOINT:=}"
: "${HORM_MODEL:=eqv2}"
: "${HORM_CONDA_ENV:=/home/wuping/.conda/envs/Hessian}"
: "${HORM_PYTHON_BIN:=${HORM_CONDA_ENV}/bin/python}"
: "${HORM_DEVICE:=cuda}"
: "${HORM_USE_DAEMON:=0}"
: "${HORM_DAEMON_SOCKET:=/tmp/horm_g16_${USER}.sock}"
: "${HORM_DAEMON_LOG:=${SCRIPT_DIR}/horm_daemon.log}"
# 优化每一步都会调用本脚本；默认安静。调试时：export HORM_VERBOSE=1
: "${HORM_VERBOSE:=0}"

# 避免 OpenMP 与 SLURM 核数不匹配时过度订阅（可按作业改）
if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ -z "${OMP_NUM_THREADS:-}" ]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
fi

if [ ! -f "${SCRIPT_DIR}/horm_external.py" ]; then
  echo "错误：未找到 horm_external.py：${SCRIPT_DIR}/horm_external.py" >&2
  exit 1
fi

if [ ! -d "${HORM_ROOT}" ]; then
  echo "错误：HORM_ROOT 目录不存在：${HORM_ROOT}" >&2
  exit 1
fi

if [ ! -x "${HORM_PYTHON_BIN}" ]; then
  echo "错误：HORM_PYTHON_BIN 不可执行：${HORM_PYTHON_BIN}（可设置 HORM_CONDA_ENV 或 HORM_PYTHON_BIN）" >&2
  exit 1
fi

if [ ! -f "${EIN_FILE}" ]; then
  echo "错误：Gaussian 传入的 EIn 不存在：${EIN_FILE}" >&2
  exit 1
fi

export HORM_ROOT HORM_CHECKPOINT HORM_MODEL HORM_DEVICE HORM_DAEMON_SOCKET HORM_DAEMON_LOG

if [ "${HORM_VERBOSE}" = "1" ]; then
  echo "调用 HORM external（HORM_PYTHON_BIN=${HORM_PYTHON_BIN}，HORM_DEVICE=${HORM_DEVICE}，HORM_MODEL=${HORM_MODEL}${HORM_CHECKPOINT:+，HORM_CHECKPOINT=${HORM_CHECKPOINT}}，HORM_USE_DAEMON=${HORM_USE_DAEMON}）" >&2
  echo "EIn=${EIN_FILE} EOu=${EOU_FILE}" >&2
  echo "Python 报错会写入：${ERR_LOG}" >&2
fi

set +e
if [ "${HORM_USE_DAEMON}" = "1" ]; then
  _daemon_ready() {
    env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
      "${HORM_PYTHON_BIN}" -c "import socket; s=socket.socket(socket.AF_UNIX); s.connect('${HORM_DAEMON_SOCKET}'); s.close()" 2>/dev/null
  }
  if ! _daemon_ready; then
    rm -f "${HORM_DAEMON_SOCKET}"
    nohup env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
      "${HORM_PYTHON_BIN}" "${SCRIPT_DIR}/horm_external.py" --daemon --socket "${HORM_DAEMON_SOCKET}" >>"${HORM_DAEMON_LOG}" 2>&1 &
    _ok=0
    for _i in $(seq 1 1200); do
      sleep 0.25
      if _daemon_ready; then
        _ok=1
        break
      fi
    done
    if [ "${_ok}" -ne 1 ]; then
      echo "错误：HORM daemon 未在 ${HORM_DAEMON_SOCKET} 就绪（见 ${HORM_DAEMON_LOG}）" >&2
      tail -n 80 "${HORM_DAEMON_LOG}" 2>/dev/null || true
      PY_EXIT=1
    else
      env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
        "${HORM_PYTHON_BIN}" "${SCRIPT_DIR}/horm_external.py" --client --socket "${HORM_DAEMON_SOCKET}" "${EIN_FILE}" "${EOU_FILE}" 2>>"${ERR_LOG}"
      PY_EXIT=$?
    fi
  else
    env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
      "${HORM_PYTHON_BIN}" "${SCRIPT_DIR}/horm_external.py" --client --socket "${HORM_DAEMON_SOCKET}" "${EIN_FILE}" "${EOU_FILE}" 2>>"${ERR_LOG}"
    PY_EXIT=$?
  fi
else
  env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
    "${HORM_PYTHON_BIN}" "${SCRIPT_DIR}/horm_external.py" "${EIN_FILE}" "${EOU_FILE}" 2>>"${ERR_LOG}"
  PY_EXIT=$?
fi
set -e

if [ "${PY_EXIT}" -ne 0 ]; then
  echo "错误：horm_external.py 退出码 ${PY_EXIT}，Gaussian 需要成功生成 .EOu。Traceback：" >&2
  cat "${ERR_LOG}" >&2
  exit 1
fi

if [ ! -f "${EOU_FILE}" ]; then
  echo "错误：未生成 ${EOU_FILE}（horm_external 异常退出或未写文件）" >&2
  cat "${ERR_LOG}" >&2
  exit 1
fi

if [ "${HORM_VERBOSE}" = "1" ]; then
  echo "HORM external 结束" >&2
fi
