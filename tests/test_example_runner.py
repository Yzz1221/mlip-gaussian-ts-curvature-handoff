import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.run import TEMPLATES, kabsch_row, read_xyz, render_input
from mlip_gaussian_handoff.gaussian_io import parse_gaussian_input


class ExampleRunnerTests(unittest.TestCase):
    def test_proper_rotation_matches_checkpoint_frame(self):
        source = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.1, 1.2, 0.0],
            [0.2, 0.3, 1.4],
        ])
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        target = source @ rotation + np.array([2.0, -3.0, 0.5])
        fitted, rmsd = kabsch_row(source, target)
        self.assertLess(rmsd, 1.0e-12)
        self.assertTrue(np.allclose(fitted, rotation, atol=1.0e-12))
        self.assertGreater(np.linalg.det(fitted), 0.0)

    def test_custom_xyz_replaces_template_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "custom.xyz"
            source.write_text("3\ncustom input\nH 0 0 0\nO 0 0 1\nH 1 0 0\n")
            numbers, coordinates, symbols = read_xyz(source)
            output = root / "opt_freq.gjf"
            render_input(
                TEMPLATES / "gaussian_calcfc" / "opt_freq.gjf",
                output, 2, numbers, coordinates, symbols, 1, 2,
            )
            geometry = parse_gaussian_input(output)
            self.assertEqual((geometry.charge, geometry.multiplicity), (1, 2))
            self.assertEqual(geometry.atomic_numbers.tolist(), [1, 8, 1])
            self.assertTrue(np.allclose(geometry.coordinates_angstrom, coordinates))
            self.assertIn("%nprocshared=2", output.read_text())


if __name__ == "__main__":
    unittest.main()
