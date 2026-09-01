from pathlib import Path
import tempfile
import unittest

import numpy as np

from mlip_gaussian_handoff.gaussian_io import (
    inject_cartesian_force_constants,
    parse_gaussian_input,
    read_fchk_array,
)


class GaussianIoTests(unittest.TestCase):
    def test_parse_gaussian_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.gjf"
            path.write_text(
                "%chk=x.chk\n#p hf/sto-3g sp\n\nTitle\n\n0 1\nC 0 0 0\nH 0 0 1\n\n",
                encoding="utf-8",
            )
            geometry = parse_gaussian_input(path)
            self.assertEqual(geometry.charge, 0)
            self.assertEqual(geometry.multiplicity, 1)
            self.assertEqual(geometry.atomic_numbers.tolist(), [6, 1])
            self.assertEqual(geometry.coordinates_angstrom.shape, (2, 3))

    def test_inject_force_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.fchk"
            target = Path(directory) / "target.fchk"
            source.write_text(
                "Atomic numbers                             I   N=           1\n"
                "           1\n"
                "Current cartesian coordinates              R   N=           3\n"
                "  0.00000000E+00  0.00000000E+00  0.00000000E+00\n"
                "Dipole Moment                              R   N=           3\n"
                "  0.00000000E+00  0.00000000E+00  0.00000000E+00\n",
                encoding="utf-8",
            )
            inject_cartesian_force_constants(source, target, np.diag([1.0, 2.0, 3.0]))
            values = read_fchk_array(target, "Cartesian Force Constants")
            self.assertEqual(values.tolist(), [1.0, 0.0, 2.0, 0.0, 0.0, 3.0])
