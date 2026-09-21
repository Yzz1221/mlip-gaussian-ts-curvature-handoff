import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np

from examples.run import SAMPLE_DATA, kabsch_row, read_xyz, render_input
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
                SAMPLE_DATA / "Gaussian_calcfc" / "rxn9" / "TS+Freq" / "opt+freq.gjf",
                output, 2, numbers, coordinates, symbols, 1, 2,
            )
            geometry = parse_gaussian_input(output)
            self.assertEqual((geometry.charge, geometry.multiplicity), (1, 2))
            self.assertEqual(geometry.atomic_numbers.tolist(), [1, 8, 1])
            self.assertTrue(np.allclose(geometry.coordinates_angstrom, coordinates))
            self.assertIn("%nprocshared=2", output.read_text())

    def test_packaged_external_input_keeps_charge_and_resolves_wrapper(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            subprocess.run([
                sys.executable, str(root / "examples/run.py"),
                "--workflow", "external_calcall", "--dataset", "gsm",
                "--reaction", "rxn9", "--output", str(output), "--dry-run",
            ], cwd=root, check=True, capture_output=True, text=True)
            geometry = parse_gaussian_input(output / "opt_freq.gjf")
            self.assertEqual((geometry.charge, geometry.multiplicity), (0, 1))
            self.assertIn("External='./horm.sh'", (output / "opt_freq.gjf").read_text())
            self.assertIn("%oldchk=ts_freq.chk", (output / "irc.gjf").read_text())
            self.assertNotIn("Guess=Read", (output / "irc.gjf").read_text())

    def test_five_reaction_example_uses_example_tree(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            subprocess.run([
                sys.executable, str(root / "examples/run.py"),
                "--example", "--workflow", "oneshot", "--dataset", "react_ot",
                "--reaction", "rxn53", "--output", str(output), "--dry-run",
            ], cwd=root, check=True, capture_output=True, text=True)
            self.assertEqual(len(list(output.glob("*.gjf"))), 3)
            self.assertIn("%oldchk=mlip_readfc.chk", (output / "irc.gjf").read_text())
            self.assertEqual(parse_gaussian_input(output / "initial_sp.gjf").charge, 0)


if __name__ == "__main__":
    unittest.main()
