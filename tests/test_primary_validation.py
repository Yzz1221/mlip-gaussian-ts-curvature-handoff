import tempfile
import unittest
from pathlib import Path

from analysis.audit_ts_freq_logs import classify
from analysis.validate import analyze


OPT_FREQ_LOG = """
 Maximum Force            0.000001  0.000450     YES
 RMS     Force            0.000001  0.000300     YES
 Maximum Displacement     0.000001  0.001800     YES
 RMS     Displacement     0.000001  0.001200     YES
 Optimization completed.
 Stationary point found.
 Normal termination of Gaussian 16
 Link1:  Proceeding to internal job step number 2.
 Harmonic frequencies (cm**-1)
 Frequencies --  -500.0  100.0  200.0
 Normal termination of Gaussian 16
"""


class PrimaryValidationTests(unittest.TestCase):
    def test_primary_minus_10_rejects_second_imaginary_mode(self):
        passed = classify(OPT_FREQ_LOG)
        self.assertEqual(passed["paper_ok_minus_10"], 1)
        changed = OPT_FREQ_LOG.replace("-500.0  100.0  200.0", "-500.0  -11.0  200.0")
        failed = classify(changed)
        self.assertEqual(failed["paper_ok_minus_10"], 0)
        self.assertEqual(failed["imaginary_below_minus_10"], 2)

    def test_missing_runs_remain_in_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rxn9").mkdir()
            (root / "rxn9" / "opt_freq.log").write_text(OPT_FREQ_LOG)
            summary = analyze("gsm", root, ["rxn9", "rxn26"])
            self.assertEqual(summary["denominator"], 2)
            self.assertEqual(summary["opt_freq_success"], 1)
            self.assertEqual(summary["opt_freq_success_rate"], 0.5)
            self.assertEqual(summary["cases"][1]["opt_freq_reason"], "missing_or_empty_log")


if __name__ == "__main__":
    unittest.main()
