import subprocess
import sys
import unittest


class UniversalExperimentCliTest(unittest.TestCase):
    def test_package_help_lists_both_supported_datasets(self):
        result = subprocess.run(
            [sys.executable, "-m", "universal_experiment", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("census", result.stdout)
        self.assertIn("aliccp", result.stdout)
        self.assertIn("--newtask-read", result.stdout)
        self.assertIn("--stage1-read", result.stdout)

    def test_newtask_read_dispatch_keeps_stage1_flag_for_the_sub_runner(self):
        # `--stage1 <dir>` must reach the newtask runner, not be prefix-matched
        # by the top-level `--stage1-read` (regression: silent wrong dispatch).
        result = subprocess.run(
            [sys.executable, "-m", "universal_experiment", "census", "--newtask-read",
             "--stage1", "somewhere"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--newtask-read", result.stderr)
        self.assertNotIn("--stage1-read", result.stderr)


if __name__ == "__main__":
    unittest.main()
