"""Dependency checker tests; no installation or measured-data changes."""
from importlib.metadata import PackageNotFoundError
from pathlib import Path
import tempfile
import unittest

from scripts.check_steel_demo_environment import check_requirements, ROOT


class EnvironmentTests(unittest.TestCase):
    def test_demo_requirements_match_verified_environment(self):
        self.assertEqual(check_requirements(ROOT / "requirements-demo.txt"), [])

    def test_ranges_includes_markers_and_missing_packages(self):
        def installed(name):
            if name == "absent":
                raise PackageNotFoundError(name)
            return {"numpy": "2.4.4", "pandas": "2.3.3"}[name]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.txt").write_text("numpy>=1.24,<2\npandas>=2,<3\nabsent>=1\n")
            (root / "main.txt").write_text('-r base.txt\npandas==2.3.3 # pinned\nskipped>=1; python_version < "1"\n')
            errors = check_requirements(root / "main.txt", installed)
            self.assertEqual(len(errors), 2)
            self.assertTrue(any("numpy" in error and "2.4.4" in error for error in errors))
            self.assertIn("absent: not installed", errors)

    def test_recursive_or_unsupported_requirements_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text("-r requirements.txt\n")
            with self.assertRaisesRegex(ValueError, "Recursive"):
                check_requirements(path)
            path.write_text("--unknown-option\n")
            with self.assertRaises(ValueError):
                check_requirements(path)


if __name__ == "__main__":
    unittest.main()
