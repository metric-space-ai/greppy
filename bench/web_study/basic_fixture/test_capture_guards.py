"""Disk refusal must happen before any build process is launched."""
import importlib.util
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARTIFACTS = Path('/Volumes/tmp/dev-artifacts/greppy/web-efficiency')

class CaptureGuardTests(unittest.TestCase):
    def test_native_refuses_each_low_volume_before_launch(self):
        for low in [Path('/Volumes/tmp/dev-artifacts/greppy'), ROOT, Path('/private/tmp')]:
            with self.subTest(low=str(low)):
                usage = lambda path: SimpleNamespace(free=(2 if Path(path) == low else 8)*1024**3)
                with patch.dict(os.environ, {'TMPDIR':'/private/tmp'}), \
                     patch('sys.argv', ['capture', 'must-not-launch']), \
                     patch('subprocess.check_output', return_value=str(ROOT)+'\n'), \
                     patch('subprocess.Popen') as launch, \
                     patch('shutil.disk_usage', side_effect=usage):
                    with self.assertRaisesRegex(SystemExit, 'each require at least 3 GiB'):
                        runpy.run_path(str(HERE/'capture_native_workflow.py'), run_name='__main__')
                    launch.assert_not_called()

    def test_cli_refuses_each_low_volume_before_launch(self):
        spec = importlib.util.spec_from_file_location('candidate_guard', HERE/'build_cli_candidate.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for low in [ARTIFACTS, ROOT, ARTIFACTS/'tmp']:
            with self.subTest(low=str(low)), tempfile.TemporaryDirectory() as evidence:
                usage = lambda path: SimpleNamespace(free=(2 if Path(path) == low else 8)*1024**3)
                with patch.object(Path, 'is_mount', return_value=True), \
                     patch('tempfile.mkdtemp', return_value=evidence), \
                     patch('subprocess.check_output', return_value='test-head\n'), \
                     patch('subprocess.Popen') as launch, \
                     patch('shutil.disk_usage', side_effect=usage):
                    with self.assertRaisesRegex(AssertionError, 'each require 3 GiB'):
                        module.build(ROOT, ARTIFACTS, 600, test_web=True)
                    launch.assert_not_called()

if __name__ == '__main__':
    unittest.main()
