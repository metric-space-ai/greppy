#!/usr/bin/env python3
"""Bind a Microsoft-signed Greppy WinFsp driver to its unsigned fork build."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys
from typing import Any


SCHEMA = "greppy.windows-driver-contract.v1"
EXPECTED_MACHINE = 0x8664
EXPECTED_UPSTREAM_COMMIT = "ddca7bd5481857a65ba552f643b8776fd070836f"


class ContractError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pe_identity(path: pathlib.Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 0x40 or raw[:2] != b"MZ":
        raise ContractError(f"{path} is not a PE file")
    pe_offset = struct.unpack_from("<I", raw, 0x3C)[0]
    if pe_offset + 24 > len(raw) or raw[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise ContractError(f"{path} has an invalid PE header")
    machine, _, _, _, _, optional_size, _ = struct.unpack_from(
        "<HHIIIHH", raw, pe_offset + 4
    )
    optional = pe_offset + 24
    if optional + optional_size > len(raw):
        raise ContractError(f"{path} has a truncated optional header")
    magic = struct.unpack_from("<H", raw, optional)[0]
    if magic == 0x20B:
        data_directories = optional + 112
    elif magic == 0x10B:
        data_directories = optional + 96
    else:
        raise ContractError(f"{path} has unsupported PE optional-header magic 0x{magic:04x}")
    checksum_offset = optional + 64
    security_entry = data_directories + (4 * 8)
    if security_entry + 8 > optional + optional_size:
        raise ContractError(f"{path} has no PE security directory")
    certificate_offset, certificate_size = struct.unpack_from(
        "<II", raw, security_entry
    )
    if certificate_size:
        if certificate_offset == 0 or certificate_offset + certificate_size > len(raw):
            raise ContractError(f"{path} has an invalid certificate table")
    elif certificate_offset:
        raise ContractError(f"{path} has a certificate offset without a size")

    canonical = bytearray(raw)
    canonical[checksum_offset : checksum_offset + 4] = b"\0" * 4
    canonical[security_entry : security_entry + 8] = b"\0" * 8
    if certificate_size:
        del canonical[certificate_offset : certificate_offset + certificate_size]
    # Authenticode may add up to seven NUL bytes before its aligned certificate
    # table. They are not part of the fork-produced image.
    removed_padding = 0
    while canonical and canonical[-1] == 0 and removed_padding < 7:
        canonical.pop()
        removed_padding += 1

    return {
        "path": str(path.resolve()),
        "sha256": sha256_bytes(raw),
        "canonical_pe_sha256": sha256_bytes(bytes(canonical)),
        "machine": f"0x{machine:04x}",
        "optional_header_magic": f"0x{magic:04x}",
        "certificate_offset": certificate_offset,
        "certificate_size": certificate_size,
        "canonical_size": len(canonical),
    }


def load_fork_manifest(path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("upstream", {}).get("commit") != EXPECTED_UPSTREAM_COMMIT:
        raise ContractError("WinFsp fork manifest does not bind the expected upstream commit")
    patches = data.get("patches")
    if not isinstance(patches, list) or len(patches) != 2:
        raise ContractError("WinFsp fork manifest must bind exactly two patches")
    normalized = []
    for patch in patches:
        if not isinstance(patch, dict) or not patch.get("path") or not patch.get("sha256"):
            raise ContractError("WinFsp fork manifest contains an invalid patch binding")
        normalized.append({"path": patch["path"], "sha256": patch["sha256"]})
    return {
        "upstream_repository": data["upstream"]["repository"],
        "upstream_tag": data["upstream"]["tag"],
        "upstream_commit": data["upstream"]["commit"],
        "patches": normalized,
    }


def build_contract(
    unsigned_path: pathlib.Path,
    signed_path: pathlib.Path,
    fork_manifest_path: pathlib.Path,
    signer_subject: str,
    signer_thumbprint: str,
) -> dict[str, Any]:
    unsigned = pe_identity(unsigned_path)
    signed = pe_identity(signed_path)
    if unsigned["machine"] != f"0x{EXPECTED_MACHINE:04x}" or signed["machine"] != f"0x{EXPECTED_MACHINE:04x}":
        raise ContractError("driver contract supports only x86_64 PE images")
    if unsigned["certificate_size"] != 0:
        raise ContractError("the fork build supplied as unsigned input already has a PE certificate")
    if signed["certificate_size"] == 0:
        raise ContractError("the Microsoft-signed driver has no embedded PE certificate")
    if unsigned["canonical_pe_sha256"] != signed["canonical_pe_sha256"]:
        raise ContractError("signed driver payload differs from the unsigned Greppy fork build")
    if not signer_subject.strip() or not signer_thumbprint.strip():
        raise ContractError("signer subject and thumbprint are required")
    return {
        "schema_version": SCHEMA,
        "architecture": "x86_64",
        "fork": load_fork_manifest(fork_manifest_path),
        "unsigned_driver": unsigned,
        "signed_driver": signed,
        "payload_identity_sha256": unsigned["canonical_pe_sha256"],
        "kernel_policy_verification": {
            "signtool_kp_verified": True,
            "signer_subject": signer_subject.strip(),
            "signer_thumbprint": signer_thumbprint.replace(" ", "").upper(),
        },
    }


def verify_contract(
    manifest_path: pathlib.Path,
    unsigned_path: pathlib.Path,
    signed_path: pathlib.Path,
    fork_manifest_path: pathlib.Path,
) -> dict[str, Any]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = expected.get("kernel_policy_verification", {})
    if expected.get("schema_version") != SCHEMA or verification.get("signtool_kp_verified") is not True:
        raise ContractError("invalid or non-kernel-verified driver contract")
    actual = build_contract(
        unsigned_path,
        signed_path,
        fork_manifest_path,
        str(verification.get("signer_subject", "")),
        str(verification.get("signer_thumbprint", "")),
    )
    if expected != actual:
        raise ContractError("driver contract does not match the supplied artifacts")
    return actual


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    identity = commands.add_parser("pe-identity")
    identity.add_argument("path", type=pathlib.Path)
    create = commands.add_parser("create")
    verify = commands.add_parser("verify")
    for command in (create, verify):
        command.add_argument("--unsigned", required=True, type=pathlib.Path)
        command.add_argument("--signed", required=True, type=pathlib.Path)
        command.add_argument("--fork-manifest", required=True, type=pathlib.Path)
    create.add_argument("--signer-subject", required=True)
    create.add_argument("--signer-thumbprint", required=True)
    create.add_argument("--output", required=True, type=pathlib.Path)
    verify.add_argument("--manifest", required=True, type=pathlib.Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "pe-identity":
            value = pe_identity(args.path)
        elif args.command == "create":
            value = build_contract(
                args.unsigned,
                args.signed,
                args.fork_manifest,
                args.signer_subject,
                args.signer_thumbprint,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            value = verify_contract(
                args.manifest, args.unsigned, args.signed, args.fork_manifest
            )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except (ContractError, OSError, json.JSONDecodeError) as error:
        print(f"windows-driver-contract: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
