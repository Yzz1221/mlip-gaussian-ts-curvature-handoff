import unittest

import numpy as np

from mlip_gaussian_handoff.units import (
    HESSIAN_EV_ANG2_TO_HARTREE_BOHR2,
    force_to_gaussian_gradient,
    hessian_to_gaussian,
)


class UnitTests(unittest.TestCase):
    def test_hessian_is_symmetrized_and_converted(self):
        raw = np.array([[1.0, 2.0], [4.0, 3.0]])
        result = hessian_to_gaussian(raw)
        expected = np.array([[1.0, 3.0], [3.0, 3.0]]) * HESSIAN_EV_ANG2_TO_HARTREE_BOHR2
        self.assertTrue(np.allclose(result, expected))

    def test_force_is_negated_to_gradient(self):
        force = np.array([[1.0, -2.0, 0.5]])
        gradient = force_to_gaussian_gradient(force)
        self.assertLess(gradient[0, 0], 0)
        self.assertGreater(gradient[0, 1], 0)
