import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class WindowsMsiTests(unittest.TestCase):
    def test_msi_has_private_driver_and_machine_scope(self):
        source = (ROOT / "platform/windows/Greppy.wxs").read_text(encoding="utf-8")
        self.assertIn('Scope="perMachine"', source)
        self.assertIn('Name="greppyworkspacefsp-x64.dll"', source)
        self.assertIn('Name="greppyworkspacefsp-x64.sys"', source)
        self.assertIn('SelfRegCost="1"', source)
        self.assertNotIn("winfsp-x64", source)
        self.assertIn('Permanent="no"', source)
        self.assertIn('System="yes"', source)
        guids = re.findall(r'\{[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}\}', source)
        self.assertEqual(len(guids), len(set(guids)))

    def test_builder_pins_wix_and_fails_closed_for_release(self):
        source = (ROOT / "tools/build_windows_msi.ps1").read_text(encoding="utf-8")
        self.assertIn("$WixVersion = '5.0.2'", source)
        self.assertIn(
            "$WixPackageSha256 = 'f30ef0c74e2a986126539c5780be93ac24e8136eaf723b1937b26272703ae173'",
            source,
        )
        self.assertIn("signtool verify /kp /all /v", source)
        self.assertIn("windows_driver_contract.py') verify", source)
        self.assertIn("wix msi validate", source)
        self.assertIn("-AllowUnsignedForSmokeTest", (
            ROOT / ".github/workflows/filesystem-cow.yml"
        ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
