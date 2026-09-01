import unittest

from mlip_gaussian_handoff.audit import classify_opt_freq


class AuditTests(unittest.TestCase):
    def test_empty_log_fails_closed(self):
        result = classify_opt_freq("")
        self.assertEqual(result["paper_ok_minus_10"], 0)
        self.assertEqual(result["paper_reason"], "missing_or_empty_log")

    def test_minimal_link1_success(self):
        text = """
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
        result = classify_opt_freq(text)
        self.assertEqual(result["paper_ok_minus_10"], 1)
        self.assertEqual(result["imaginary_below_minus_10"], 1)
