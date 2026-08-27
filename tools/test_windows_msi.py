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
        self.assertIn('Root="HKLM"', source)
        self.assertIn('Key="Software\\Microsoft\\Windows\\CurrentVersion\\Run"', source)
        self.assertIn('Name="GreppyWorkspaceProvider"', source)
        self.assertIn(
            'Value="&quot;[INSTALLFOLDER]greppy.exe&quot; workspace setup"',
            source,
        )
        guids = re.findall(r'\{[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}\}', source)
        self.assertEqual(len(guids), len(set(guids)))

    def test_builder_pins_wix_and_fails_closed_for_release(self):
        source = (ROOT / "tools/build_windows_msi.ps1").read_text(encoding="utf-8")
        self.assertIn("$WixVersion = '5.0.2'", source)
        self.assertIn(
            "$WixPackageSha256 = 'f30ef0c74e2a986126539c5780be93ac24e8136eaf723b1937b26272703ae173'",
            source,
        )
        self.assertIn("verify_windows_driver_signatures.ps1", source)
        self.assertIn("-DriverCatalogPath", source)
        self.assertIn("greppy-windows-driver-signature-evidence.json", source)
        self.assertIn("windows_driver_contract.py') verify", source)
        self.assertIn("'release_daemon_stress.ps1'", source)
        self.assertIn("wix msi validate", source)
        workflow = (ROOT / ".github/workflows/filesystem-cow.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("-AllowUnsignedForSmokeTest", workflow)
        self.assertIn(r"target\debug\greppy.exe", workflow)
        self.assertIn("Language.Parser]::ParseFile", workflow)
        self.assertIn("bench\\release_daemon_stress.ps1", workflow)
        self.assertIn("verify_windows_driver_signatures.ps1 $dist", workflow)
        self.assertNotIn(
            "cargo build -p greppy --bin greppy --release --locked\n"
            "          --no-default-features\n"
            "          --features ci-test-assets,cpu-only,bash-smart",
            workflow,
        )

    def test_driver_signature_gate_rejects_attestation(self):
        source = (
            ROOT / "tools/verify_windows_driver_signatures.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("signtool verify /kp /all /v", source)
        self.assertIn("Get-AuthenticodeSignature", source)
        self.assertIn("1.3.6.1.4.1.311.10.3.5'", source)
        self.assertIn("1.3.6.1.4.1.311.10.3.5.1'", source)
        self.assertIn("attestation-signed driver is not release eligible", source)
        self.assertIn("signature_class = 'hlk-dashboard'", source)


if __name__ == "__main__":
    unittest.main()
