"""Offline installer integration checks: no downloads, packages or inference."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "install with spaces"
        self.log = self.base / "calls"
        fake = self.base / "python"
        fake.write_text(r'''#!/usr/bin/env python3
import os, pathlib, sys
if sys.argv[1] == '-c':
    sys.exit(0)
root = pathlib.Path(sys.argv[3]) / 'bin'
root.mkdir(parents=True)
for name in ('python', 'ygaraty'):
    target = root / name
    target.write_text("#!/usr/bin/env python3\nimport json, os, sys\nwith open(os.environ['INSTALL_TEST_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + chr(10))\nif '-m' in sys.argv and os.environ.get('FAIL_PIP'): sys.exit(9)\n")
    target.chmod(0o755)
''')
        fake.chmod(0o755)
        self.env = {
            **os.environ,
            "YGARATY_INSTALL_DIR": str(self.root),
            "YGARATY_SOURCE_DIR": str(INSTALLER.parent),
            "PYTHON": str(fake),
            "INSTALL_TEST_LOG": str(self.log),
        }

    def run_installer(self, *args, pipe=False):
        command = ["bash", "-s", "--"] if pipe else ["bash", str(INSTALLER)]
        return subprocess.run(
            [*command, *args], env=self.env, text=True,
            input=INSTALLER.read_text() if pipe else None,
            capture_output=True,
        )

    def test_install_and_reuse_via_stdin(self):
        result = self.run_installer('--install-only', pipe=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / '.installed').is_file())
        calls = self.log.read_text()
        self.assertIn('pip', calls)
        result = self.run_installer('--num-workers', '2', '--checkpoint', '/model with spaces', '--port', '9000')
        self.assertEqual(result.returncode, 0, result.stderr)
        new_calls = self.log.read_text()[len(calls):]
        self.assertNotIn('pip', new_calls)
        self.assertIn('launch', new_calls)
        self.assertIn('/model with spaces', new_calls)
        self.assertIn('9000', new_calls)

    def test_failed_install_never_launches_or_marks_complete(self):
        self.env['FAIL_PIP'] = '1'
        result = self.run_installer('--num-workers', '2', '--checkpoint', '/model')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / '.installed').exists())
        self.assertNotIn('launch', self.log.read_text())
        result = self.run_installer('--install-only')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already exists', result.stderr)

    def test_remote_source_uses_public_repo_and_requested_ref(self):
        self.env.pop('YGARATY_SOURCE_DIR')
        self.env['YGARATY_REF'] = 'v0.1.0'
        git = self.base / 'git'
        git.write_text(
            '#!/usr/bin/env python3\n'
            'import json, os, sys\n'
            "with open(os.environ['INSTALL_TEST_LOG'], 'a') as f: "
            'f.write(json.dumps(sys.argv[1:]) + chr(10))\n'
        )
        git.chmod(0o755)
        self.env['PATH'] = str(self.base) + os.pathsep + self.env['PATH']
        result = self.run_installer('--install-only', pipe=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text()
        self.assertIn('https://github.com/ygaraty/ygaraty.git', calls)
        self.assertIn('"fetch", "--depth", "1", "origin", "v0.1.0"', calls)
        self.assertIn('"checkout", "-q", "--detach", "FETCH_HEAD"', calls)

    def test_persistent_path_guidance_and_version_mismatch(self):
        result = self.run_installer('--install-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('export PATH=', result.stdout)
        self.assertIn('~/.zshrc', result.stdout)
        self.assertIn('~/.bashrc', result.stdout)
        self.env['YGARATY_REF'] = 'another-version'
        result = self.run_installer('--install-only')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Installed version is', result.stderr)

    def test_help_does_not_install(self):
        result = self.run_installer('--help')
        self.assertEqual(result.returncode, 0)
        self.assertFalse(self.root.exists())

    def test_relative_install_path_refused(self):
        self.env['YGARATY_INSTALL_DIR'] = 'relative'
        result = self.run_installer('--install-only')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('absolute path', result.stderr)


    def test_no_arguments_refuses_before_installation(self):
        result = self.run_installer(pipe=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--num-workers is required', result.stderr)
        self.assertFalse(self.root.exists())

    def test_worker_count_is_forwarded(self):
        import json
        result = self.run_installer('--num-workers', '2', pipe=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(calls[-1], ["launch", "--num-workers", "2"])


if __name__ == '__main__':
    unittest.main()
