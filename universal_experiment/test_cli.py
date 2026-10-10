import subprocess
import sys
import unittest


class UniversalExperimentCliTest(unittest.TestCase):
    def test_package_help_describes_paired_b_u_protocol(self):
        result = subprocess.run(
            [sys.executable, "-m", "universal_experiment", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("census", result.stdout)
        self.assertIn("aliccp", result.stdout)
        self.assertIn("--read-policy", result.stdout)
        self.assertIn("--u-mode", result.stdout)


if __name__ == "__main__":
    unittest.main()
