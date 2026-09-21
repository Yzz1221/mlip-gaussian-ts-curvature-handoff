#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


NORMAL_RE = re.compile(r"Normal termination of Gaussian", re.IGNORECASE)
ERROR_RE = re.compile(r"Error termination|Abnormal termination", re.IGNORECASE)
FREQUENCY_RE = re.compile(r"^\s*Frequencies\s+--\s+(.*)$")
SECOND_STEP_RE = re.compile(
    r"Link1:\s+Proceeding to internal job step number\s+2\."
)

# Gaussian 会把一条 ``Opt ... Freq`` 路由展开为两个内部步骤：先进行几何优化，
# 再在优化后的几何上计算频率。只含 Freq 的步骤可能再次打印 Berny 收敛表，
# 甚至再次打印 ``Optimization completed``。这些后续信息只是诊断输出，
# 绝不能用来判断前面的 TS Opt 是否收敛。


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def is_four_yes_block(lines: list[str]) -> bool:
    if len(lines) != 4:
        return False
    normalized = [line.strip() for line in lines]
    return (
        "Maximum Force" in normalized[0]
        and normalized[0].endswith("YES")
        and "RMS" in normalized[1]
        and "Force" in normalized[1]
        and "Displacement" not in normalized[1]
        and normalized[1].endswith("YES")
        and "Maximum Displacement" in normalized[2]
        and normalized[2].endswith("YES")
        and "RMS" in normalized[3]
        and "Displacement" in normalized[3]
        and normalized[3].endswith("YES")
    )


def last_completed_optimization_index(lines: list[str]) -> int | None:
    """返回指定日志片段中最后一个真实的优化完成标记位置。"""
    for index in range(len(lines) - 1, -1, -1):
        if "Optimization completed" in lines[index]:
            return index
    return None


def last_completed_optimization_has_four_yes(
    lines: list[str], completion_index: int | None
) -> bool:
    """只检查紧邻 ``Optimization completed`` 之前的收敛表。

    以优化完成标记为锚点，可以防止 Freq 步骤随后打印的收敛表覆盖真实的
    Opt 收敛结果。
    """
    if completion_index is None:
        return False
    window = lines[max(0, completion_index - 40) : completion_index]
    for index in range(len(window) - 4, -1, -1):
        if is_four_yes_block(window[index : index + 4]):
            return True
    return False


def count_frequencies_below(
    text: str, harmonic_index: int, threshold_cm: float
) -> int | None:
    """只统计最后一个谐振频率区块中的频率。"""
    if harmonic_index < 0:
        return None
    count = 0
    for line in text[harmonic_index:].splitlines():
        match = FREQUENCY_RE.match(line)
        if not match:
            continue
        for token in match.group(1).split():
            try:
                value = float(token.replace("D", "E").replace("d", "e"))
            except ValueError:
                continue
            if value < threshold_cm:
                count += 1
    return count


def classify(text: str) -> dict[str, object]:
    # 如果 Link1 明确启动内部步骤 2，则边界之前是包含 Opt 的步骤，边界之后
    # 是最终的纯 Freq 步骤。如果没有显式边界，则把整份日志视为一个联合任务。
    second_step_match = SECOND_STEP_RE.search(text)
    opt_end_index = second_step_match.start() if second_step_match else len(text)
    opt_text = text[:opt_end_index]
    opt_lines = opt_text.splitlines()
    completion_line_index = last_completed_optimization_index(opt_lines)
    completion_index = opt_text.rfind("Optimization completed")
    harmonic_index = text.rfind("Harmonic frequencies (cm**-1)")
    stationary_index = (
        opt_text.find("Stationary point found", completion_index)
        if completion_index >= 0
        else -1
    )

    # 仅在日志任意位置出现一次 Normal termination 并不充分，否则 Opt 已完成、
    # 但 Freq 被截断的任务也会误判通过。对于显式 Link1 任务，要求 Opt 步骤在
    # 步骤 2 开始前正常结束；对于单一联合任务，末尾的正常结束同时结束两项操作。
    normal_positions = [match.start() for match in NORMAL_RE.finditer(text)]
    opt_stage_normal = any(
        completion_index >= 0
        and position > completion_index
        and position < opt_end_index
        for position in normal_positions
    )
    freq_stage_normal = any(
        harmonic_index >= 0 and position > harmonic_index
        for position in normal_positions
    )
    error = bool(ERROR_RE.search(text))
    if second_step_match is None:
        opt_stage_normal = freq_stage_normal and completion_index >= 0
    normal = opt_stage_normal and freq_stage_normal
    stationary_point = stationary_index >= 0
    negligible_forces_completion = (
        "Optimization completed on the basis of negligible forces" in opt_text
    )
    four_yes = last_completed_optimization_has_four_yes(
        opt_lines, completion_line_index
    )
    imaginary_below_minus_10 = count_frequencies_below(
        text, harmonic_index, -10.0
    )
    imaginary_below_minus_50 = count_frequencies_below(
        text, harmonic_index, -50.0
    )
    imaginary_below_zero = count_frequencies_below(text, harmonic_index, 0.0)

    if not text.strip():
        reason = "missing_or_empty_log"
    elif error:
        reason = "error_or_abnormal_termination"
    elif completion_index < 0:
        reason = "no_optimization_completed"
    elif not four_yes:
        reason = "last_completed_optimization_not_four_yes"
    elif not stationary_point:
        reason = "no_stationary_point_after_optimization"
    elif not opt_stage_normal:
        reason = "optimization_step_not_normally_terminated"
    elif imaginary_below_minus_10 is None:
        reason = "no_harmonic_frequency_block"
    elif not freq_stage_normal:
        reason = "frequency_step_not_normally_terminated"
    elif imaginary_below_minus_10 != 1:
        reason = f"imaginary_modes_below_minus_10_is_{imaginary_below_minus_10}"
    else:
        reason = "ok"

    paper_ok = reason == "ok"
    # 即使某个位移判据仍为 NO，Gaussian 也可能基于 negligible forces 合法接受
    # 一个驻点。这里把这种较宽的 Gaussian 原生接受标准与“四项 YES”论文标准
    # 分开保存，使判据选择可见、可比较，而不是在统计中静默改变。
    gaussian_accepted_ok = (
        normal
        and not error
        and completion_index >= 0
        and stationary_point
        and imaginary_below_minus_10 == 1
    )
    strict_ok = (
        normal
        and not error
        and four_yes
        and stationary_point
        and imaginary_below_minus_50 == 1
    )
    return {
        "normal_termination": int(normal),
        "optimization_step_normal_termination": int(opt_stage_normal),
        "frequency_step_normal_termination": int(freq_stage_normal),
        "error_or_abnormal_termination": int(error),
        "stationary_point_found": int(stationary_point),
        "optimization_completed_on_negligible_forces": int(
            negligible_forces_completion
        ),
        "last_completed_optimization_four_yes": int(four_yes),
        "imaginary_below_minus_10": imaginary_below_minus_10,
        "imaginary_below_minus_50": imaginary_below_minus_50,
        "imaginary_below_zero": imaginary_below_zero,
        "paper_ok_minus_10": int(paper_ok),
        "gaussian_accepted_ok_minus_10": int(gaussian_accepted_ok),
        "strict_ok_minus_50": int(strict_ok),
        "paper_reason": reason,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="独立审计 Gaussian TS+Freq 日志，不修改任何原始来源文件。"
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--relative-log", required=True)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    reaction_directories = sorted(
        (path for path in root.glob("rxn*") if path.is_dir()),
        key=lambda path: path.name,
    )
    for reaction_directory in reaction_directories:
        log = reaction_directory / args.relative_log
        text = read_text(log)
        row: dict[str, object] = {
            "experiment_id": args.experiment_id,
            "stem": reaction_directory.name,
            "log": str(log),
            "log_exists": int(log.is_file()),
            "log_size_bytes": log.stat().st_size if log.is_file() else 0,
        }
        row.update(classify(text))
        rows.append(row)

    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    fieldnames = list(rows[0]) if rows else ["experiment_id", "stem", "log"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reason_counts = Counter(str(row["paper_reason"]) for row in rows)
    summary = {
        "experiment_id": args.experiment_id,
        "root": str(root),
        "relative_log": args.relative_log,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": file_sha256(Path(__file__).resolve()),
        "expected_cases": len(reaction_directories),
        "logs_found": sum(int(row["log_exists"]) for row in rows),
        "normal_termination": sum(int(row["normal_termination"]) for row in rows),
        "optimization_step_normal_termination": sum(
            int(row["optimization_step_normal_termination"]) for row in rows
        ),
        "frequency_step_normal_termination": sum(
            int(row["frequency_step_normal_termination"]) for row in rows
        ),
        "stationary_point_found": sum(
            int(row["stationary_point_found"]) for row in rows
        ),
        "optimization_completed_on_negligible_forces": sum(
            int(row["optimization_completed_on_negligible_forces"])
            for row in rows
        ),
        "four_yes": sum(
            int(row["last_completed_optimization_four_yes"]) for row in rows
        ),
        "paper_ok_minus_10": sum(int(row["paper_ok_minus_10"]) for row in rows),
        "gaussian_accepted_ok_minus_10": sum(
            int(row["gaussian_accepted_ok_minus_10"]) for row in rows
        ),
        "strict_ok_minus_50": sum(int(row["strict_ok_minus_50"]) for row in rows),
        "paper_reason_counts": dict(sorted(reason_counts.items())),
        "csv": str(csv_path),
    }
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
