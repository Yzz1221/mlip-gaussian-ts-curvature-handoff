"""Link1-aware classification of Gaussian TS optimization plus frequency logs."""

from __future__ import annotations

import re

NORMAL_RE = re.compile(r"Normal termination of Gaussian", re.IGNORECASE)
ERROR_RE = re.compile(r"Error termination|Abnormal termination", re.IGNORECASE)
FREQUENCY_RE = re.compile(r"^\s*Frequencies\s+--\s+(.*)$")
SECOND_STEP_RE = re.compile(r"Link1:\s+Proceeding to internal job step number\s+2\.")


def _four_yes(lines: list[str]) -> bool:
    if len(lines) != 4:
        return False
    values = [line.strip() for line in lines]
    return (
        "Maximum Force" in values[0] and values[0].endswith("YES")
        and "RMS" in values[1] and "Force" in values[1]
        and "Displacement" not in values[1] and values[1].endswith("YES")
        and "Maximum Displacement" in values[2] and values[2].endswith("YES")
        and "RMS" in values[3] and "Displacement" in values[3]
        and values[3].endswith("YES")
    )


def _last_four_yes(lines: list[str], completion: int | None) -> bool:
    if completion is None:
        return False
    window = lines[max(0, completion - 40) : completion]
    return any(_four_yes(window[i : i + 4]) for i in range(len(window) - 4, -1, -1))


def _frequency_count(text: str, start: int, threshold: float) -> int | None:
    if start < 0:
        return None
    count = 0
    for line in text[start:].splitlines():
        match = FREQUENCY_RE.match(line)
        if match:
            for token in match.group(1).split():
                try:
                    count += float(token.replace("D", "E")) < threshold
                except ValueError:
                    pass
    return int(count)


def classify_opt_freq(text: str) -> dict[str, object]:
    second = SECOND_STEP_RE.search(text)
    opt_end = second.start() if second else len(text)
    opt_text = text[:opt_end]
    opt_lines = opt_text.splitlines()
    completion_line = next(
        (i for i in range(len(opt_lines) - 1, -1, -1) if "Optimization completed" in opt_lines[i]),
        None,
    )
    completion = opt_text.rfind("Optimization completed")
    harmonic = text.rfind("Harmonic frequencies (cm**-1)")
    stationary = completion >= 0 and opt_text.find("Stationary point found", completion) >= 0
    normal_positions = [match.start() for match in NORMAL_RE.finditer(text)]
    opt_normal = any(completion >= 0 and completion < pos < opt_end for pos in normal_positions)
    freq_normal = any(harmonic >= 0 and pos > harmonic for pos in normal_positions)
    if second is None:
        opt_normal = freq_normal and completion >= 0
    error = bool(ERROR_RE.search(text))
    four_yes = _last_four_yes(opt_lines, completion_line)
    imag10 = _frequency_count(text, harmonic, -10.0)
    imag50 = _frequency_count(text, harmonic, -50.0)
    imag0 = _frequency_count(text, harmonic, 0.0)

    if not text.strip():
        reason = "missing_or_empty_log"
    elif error:
        reason = "error_or_abnormal_termination"
    elif completion < 0:
        reason = "no_optimization_completed"
    elif not four_yes:
        reason = "last_completed_optimization_not_four_yes"
    elif not stationary:
        reason = "no_stationary_point_after_optimization"
    elif not opt_normal:
        reason = "optimization_step_not_normally_terminated"
    elif imag10 is None:
        reason = "no_harmonic_frequency_block"
    elif not freq_normal:
        reason = "frequency_step_not_normally_terminated"
    elif imag10 != 1:
        reason = f"imaginary_modes_below_minus_10_is_{imag10}"
    else:
        reason = "ok"
    normal = opt_normal and freq_normal
    return {
        "paper_ok_minus_10": int(reason == "ok"),
        "gaussian_accepted_ok_minus_10": int(
            normal and not error and completion >= 0 and stationary and imag10 == 1
        ),
        "strict_ok_minus_50": int(
            normal and not error and four_yes and stationary and imag50 == 1
        ),
        "optimization_step_normal_termination": int(opt_normal),
        "frequency_step_normal_termination": int(freq_normal),
        "stationary_point_found": int(stationary),
        "last_completed_optimization_four_yes": int(four_yes),
        "imaginary_below_minus_10": imag10,
        "imaginary_below_minus_50": imag50,
        "imaginary_below_zero": imag0,
        "paper_reason": reason,
    }
