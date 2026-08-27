import json
import pathlib
import struct
import tempfile
import unittest

from tools import windows_driver_contract as contract


def fake_pe(certificate: bytes = b"") -> bytes:
    raw = bytearray(512)
    raw[:2] = b"MZ"
    struct.pack_into("<I", raw, 0x3C, 0x80)
    raw[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", raw, 0x84, 0x8664, 1, 0, 0, 0, 240, 0x2022)
    optional = 0x98
    struct.pack_into("<H", raw, optional, 0x20B)
    struct.pack_into("<I", raw, optional + 64, 0x12345678)
    if certificate:
        offset = len(raw)
        struct.pack_into("<II", raw, optional + 112 + (4 * 8), offset, len(certificate))
        raw.extend(certificate)
    return bytes(raw)


class WindowsDriverContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.fork = self.root / "upstream.json"
        self.fork.write_text(
            json.dumps(
                {
                    "schema": contract.EXPECTED_FORK_SCHEMA,
                    "repository": contract.EXPECTED_UPSTREAM_REPOSITORY,
                    "tag": contract.EXPECTED_UPSTREAM_TAG,
                    "commit": contract.EXPECTED_UPSTREAM_COMMIT,
                    "patches": [
                        {"path": "patches/0001.patch", "sha256": "1" * 64},
                        {"path": "patches/0002.patch", "sha256": "2" * 64},
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.unsigned = self.root / "unsigned.sys"
        self.signed = self.root / "signed.sys"
        self.catalog = self.root / "greppyworkspacefsp-x64.cat"
        self.unsigned.write_bytes(fake_pe())
        self.signed.write_bytes(fake_pe(b"signed!!"))
        self.catalog.write_bytes(b"signed catalog")
        self.signature_evidence = self.root / "signature-evidence.json"
        self.write_signature_evidence()

    def write_signature_evidence(self, *, attestation=False):
        eku_oids = [contract.HLK_VERIFICATION_OID]
        if attestation:
            eku_oids.append(contract.ATTESTATION_OID)
        self.signature_evidence.write_text(
            json.dumps(
                {
                    "schema_version": contract.SIGNATURE_EVIDENCE_SCHEMA,
                    "signature_class": "hlk-dashboard",
                    "driver_sha256": contract.sha256_file(self.signed),
                    "catalog_sha256": contract.sha256_file(self.catalog),
                    "driver_signer_subject": "Microsoft Windows Hardware Compatibility Publisher",
                    "driver_signer_thumbprint": "AA",
                    "catalog_signer_subject": "Microsoft Windows Hardware Compatibility Publisher",
                    "catalog_signer_thumbprint": "BB",
                    "catalog_enhanced_key_usage_oids": eku_oids,
                    "hardware_driver_verification_oid": contract.HLK_VERIFICATION_OID,
                    "attestation_oid_absent": not attestation,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_signature_is_removed_from_payload_identity(self):
        unsigned = contract.pe_identity(self.unsigned)
        signed = contract.pe_identity(self.signed)
        self.assertEqual(unsigned["canonical_pe_sha256"], signed["canonical_pe_sha256"])
        self.assertEqual(0, unsigned["certificate_size"])
        self.assertEqual(8, signed["certificate_size"])

    def test_contract_round_trip(self):
        value = contract.build_contract(
            self.unsigned,
            self.signed,
            self.catalog,
            self.fork,
            self.signature_evidence,
        )
        manifest = self.root / "contract.json"
        manifest.write_text(json.dumps(value), encoding="utf-8")
        self.assertEqual(
            value,
            contract.verify_contract(
                manifest,
                self.unsigned,
                self.signed,
                self.catalog,
                self.fork,
                self.signature_evidence,
            ),
        )
        self.assertEqual(
            value,
            contract.verify_signed_contract(
                manifest,
                self.signed,
                self.catalog,
                self.fork,
                self.signature_evidence,
            ),
        )

    def test_repository_fork_manifest_matches_driver_contract_schema(self):
        repository_manifest = (
            pathlib.Path(__file__).resolve().parents[1]
            / "third_party"
            / "winfsp-greppy"
            / "upstream.json"
        )
        loaded = contract.load_fork_manifest(repository_manifest)
        self.assertEqual(contract.EXPECTED_UPSTREAM_COMMIT, loaded["upstream_commit"])
        self.assertEqual(2, len(loaded["patches"]))

    def test_changed_payload_is_rejected(self):
        changed = bytearray(fake_pe(b"signed!!"))
        changed[300] = 1
        self.signed.write_bytes(changed)
        with self.assertRaisesRegex(contract.ContractError, "payload differs"):
            contract.build_contract(
                self.unsigned,
                self.signed,
                self.catalog,
                self.fork,
                self.signature_evidence,
            )

    def test_unsigned_signed_artifact_is_rejected(self):
        self.signed.write_bytes(fake_pe())
        with self.assertRaisesRegex(contract.ContractError, "no embedded PE certificate"):
            contract.build_contract(
                self.unsigned,
                self.signed,
                self.catalog,
                self.fork,
                self.signature_evidence,
            )

    def test_attestation_signature_evidence_is_rejected(self):
        self.write_signature_evidence(attestation=True)
        with self.assertRaisesRegex(contract.ContractError, "attestation-signed"):
            contract.build_contract(
                self.unsigned,
                self.signed,
                self.catalog,
                self.fork,
                self.signature_evidence,
            )


if __name__ == "__main__":
    unittest.main()
