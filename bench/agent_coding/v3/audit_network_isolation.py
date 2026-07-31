#!/usr/bin/env python3
"""Audit the preregistered Docker-only network boundary for V3 agents."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


POLICY_SCHEMA = "greppy.agent-coding-v3.docker-network-policy.1"
REPORT_SCHEMA = "greppy.provider-only-egress.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
URL_RE = re.compile(r"https://[A-Za-z0-9.-]+(?::\d+)?(?:/[A-Za-z0-9._~!$&'()*+,;=:@%/-]*)?")
UTC = dt.timezone.utc


class AuditError(ValueError):
    """A policy or Docker invariant failed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_version(text: str) -> tuple[int, ...] | None:
    match = re.search(r"(?<!\d)(\d+)(?:\.(\d+))?(?:\.(\d+))?", text)
    return tuple(int(value) for value in match.groups(default="0")) if match else None


def version_at_least(observed: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    width = max(len(observed), len(minimum))
    return observed + (0,) * (width - len(observed)) >= minimum + (0,) * (width - len(minimum))


def load_policy(path: Path) -> dict[str, Any]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read policy: {exc}") from exc
    if not isinstance(policy, dict) or policy.get("schema_version") != POLICY_SCHEMA:
        raise AuditError("unsupported network policy schema")
    if policy.get("status") != "sealed":
        raise AuditError("network policy status must be 'sealed'; templates cannot pass")
    for section in ("docker", "topology", "provider_contract", "negative_probes", "mount_isolation"):
        if not isinstance(policy.get(section), dict):
            raise AuditError(f"policy section {section} must be an object")
    return policy


def resolve_policy_path(config_path: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AuditError(f"{field} must be a path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def provider_hosts_from_source(source: bytes) -> set[tuple[str, int]]:
    hosts: set[tuple[str, int]] = set()
    for raw in URL_RE.findall(source.decode("utf-8", "replace")):
        parsed = urlparse(raw)
        if parsed.scheme == "https" and parsed.hostname:
            hosts.add((parsed.hostname.lower(), parsed.port or 443))
    return hosts


def validate_provider_contract(config_path: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    contract = policy["provider_contract"]
    source_path = resolve_policy_path(config_path, contract.get("provider_source"), "provider_source")
    try:
        source = source_path.read_bytes()
    except OSError as exc:
        raise AuditError(f"cannot read frozen provider source: {exc}") from exc
    source_hash = digest(source)
    expected_source_hash = contract.get("provider_source_sha256")
    if not isinstance(expected_source_hash, str) or HEX64.fullmatch(expected_source_hash) is None or source_hash != expected_source_hash:
        raise AuditError("provider source SHA-256 does not match preregistration")
    allow = contract.get("allow_connect")
    if not isinstance(allow, list) or not allow:
        raise AuditError("allow_connect must be nonempty")
    allow_pairs: set[tuple[str, int]] = set()
    for row in allow:
        if not isinstance(row, dict) or not isinstance(row.get("host"), str) or not isinstance(row.get("port"), int):
            raise AuditError("allow_connect rows need host and integer port")
        host = row["host"].lower().rstrip(".")
        port = row["port"]
        if port != 443 or host == "github.com" or host.endswith(".github.com"):
            raise AuditError("only frozen MiniMax HTTPS provider hosts are allowed")
        allow_pairs.add((host, port))
    source_pairs = provider_hosts_from_source(source)
    if allow_pairs != source_pairs:
        raise AuditError(f"CONNECT allowlist {sorted(allow_pairs)} does not equal provider source hosts {sorted(source_pairs)}")
    probe = urlparse(str(contract.get("probe_url", "")))
    if (probe.hostname, probe.port or 443) not in allow_pairs or probe.scheme != "https":
        raise AuditError("provider probe URL is outside the frozen allowlist")
    policy_hash = contract.get("proxy_policy_sha256")
    computed_policy_hash = digest(canonical_json({"allow_connect": contract["allow_connect"]}))
    if not isinstance(policy_hash, str) or HEX64.fullmatch(policy_hash) is None or policy_hash != computed_policy_hash:
        raise AuditError("proxy_policy_sha256 does not bind the canonical CONNECT allowlist")
    return {
        "provider_source_sha256": source_hash,
        "provider_hosts": [{"host": host, "port": port} for host, port in sorted(allow_pairs)],
        "proxy_policy_sha256": policy_hash,
        "probe_url": contract["probe_url"],
    }


def docker_json(binary: str, args: Sequence[str], timeout: int = 30) -> Any:
    proc = subprocess.run([binary, *args], capture_output=True, text=True, errors="replace", timeout=timeout)
    if proc.returncode != 0:
        raise AuditError(f"docker {' '.join(args[:4])} failed: {proc.stderr.strip()[-500:]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AuditError(f"docker {' '.join(args[:4])} returned invalid JSON") from exc


def inspect_one(binary: str, kind: str, name: str) -> dict[str, Any]:
    value = docker_json(binary, [kind, "inspect", name])
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise AuditError(f"docker {kind} inspect {name} did not return one object")
    return value[0]


def forbidden_mount_roots(policy: Mapping[str, Any]) -> list[Path]:
    isolation = policy["mount_isolation"]
    roots: list[Path] = []
    for name in isolation.get("forbidden_root_env", []):
        if not isinstance(name, str) or not os.environ.get(name):
            raise AuditError(f"required forbidden-root environment {name!r} is unset")
        roots.append(Path(os.environ[name]).expanduser().resolve())
    nvme = os.environ.get("GREPPY_BENCH_NVME_ROOT")
    relative = isolation.get("forbidden_nvme_relative_paths", [])
    if relative and not nvme:
        raise AuditError("GREPPY_BENCH_NVME_ROOT is required to protect builder paths")
    for value in relative:
        if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
            raise AuditError("forbidden NVMe paths must be safe relative paths")
        roots.append((Path(nvme).expanduser().resolve() / value).resolve())
    return roots


def path_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_topology(binary: str, policy: Mapping[str, Any], provider: Mapping[str, Any]) -> dict[str, Any]:
    topology = policy["topology"]
    internal_name = topology.get("agent_internal_network")
    egress_name = topology.get("proxy_egress_network")
    proxy_name = topology.get("proxy_container")
    if not all(isinstance(value, str) and value for value in (internal_name, egress_name, proxy_name)) or internal_name == egress_name:
        raise AuditError("topology names must be distinct nonempty strings")
    internal = inspect_one(binary, "network", internal_name)
    egress = inspect_one(binary, "network", egress_name)
    if internal.get("Internal") is not True:
        raise AuditError("agent network is not Docker-internal")
    if egress.get("Internal") is True:
        raise AuditError("proxy egress network must not be internal")
    proxy = inspect_one(binary, "container", proxy_name)
    if proxy.get("State", {}).get("Running") is not True:
        raise AuditError("allowlist proxy is not running")
    networks = proxy.get("NetworkSettings", {}).get("Networks", {})
    if set(networks) != {internal_name, egress_name}:
        raise AuditError("allowlist proxy must be dual-homed on exactly internal and egress networks")
    proxy_ip = networks[internal_name].get("IPAddress")
    if not isinstance(proxy_ip, str) or not proxy_ip:
        raise AuditError("proxy has no address on the agent-internal network")
    expected_image = topology.get("proxy_image_id")
    if not isinstance(expected_image, str) or not expected_image.startswith("sha256:") or proxy.get("Image") != expected_image:
        raise AuditError("running proxy image ID does not equal preregistration")
    labels = proxy.get("Config", {}).get("Labels") or {}
    if labels.get("dev.greppy.v3.role") != "allowlist-connect-proxy" or labels.get("dev.greppy.v3.proxy-policy-sha256") != provider["proxy_policy_sha256"]:
        raise AuditError("proxy role/policy labels do not bind the frozen CONNECT policy")
    policy_path = topology.get("proxy_policy_path")
    if not isinstance(policy_path, str) or not policy_path.startswith("/"):
        raise AuditError("proxy_policy_path must be an absolute in-container path")
    policy_read = subprocess.run(
        [binary, "exec", proxy_name, "cat", policy_path], capture_output=True,
        timeout=20,
    )
    if policy_read.returncode != 0 or digest(policy_read.stdout) != provider["proxy_policy_sha256"]:
        raise AuditError("baked proxy policy bytes do not match the frozen allowlist hash")
    try:
        baked_policy = json.loads(policy_read.stdout)
    except json.JSONDecodeError as exc:
        raise AuditError("baked proxy policy is not JSON") from exc
    if digest(canonical_json(baked_policy)) != provider["proxy_policy_sha256"]:
        raise AuditError("baked proxy policy is not the canonical frozen allowlist")
    mounts = proxy.get("Mounts") or []
    allowed_mounts = policy["mount_isolation"].get("proxy_allowed_mounts", [])
    if allowed_mounts != []:
        raise AuditError("V3 proxy image must bake its policy and have no host mounts")
    actual_sources = [Path(row["Source"]).resolve() for row in mounts if isinstance(row, dict) and isinstance(row.get("Source"), str)]
    if actual_sources:
        raise AuditError("V3 proxy container must have no host mounts")
    protected = forbidden_mount_roots(policy)
    if any(path_under(source, root) for source in actual_sources for root in protected):
        raise AuditError("proxy can read NAS or builder materials")
    return {
        "internal_network": {"name": internal_name, "id": internal.get("Id"), "internal": True},
        "egress_network": {"name": egress_name, "id": egress.get("Id"), "internal": False},
        "proxy": {"name": proxy_name, "id": proxy.get("Id"), "image_id": proxy.get("Image"), "networks": sorted(networks), "mount_count": len(mounts)},
        "proxy_internal_ip": proxy_ip,
    }


PROBE_PROGRAM = r'''
import json, socket, sys, urllib.error, urllib.request
c=json.loads(sys.argv[1]); out={"provider":{},"proxy_denied":[],"dns_denied":[],"direct_denied":[]}; ok=True
proxy=urllib.request.ProxyHandler({"http":c["proxy"],"https":c["proxy"]}); opener=urllib.request.build_opener(proxy)
try:
    with opener.open(c["provider_url"], timeout=c["timeout"]) as r: code=r.status
except urllib.error.HTTPError as e: code=e.code
except Exception as e: code=None; out["provider"]["error"]=type(e).__name__
out["provider"].update({"reachable":code in c["provider_statuses"],"status":code}); ok &= out["provider"]["reachable"]
for url in c["proxy_denied_urls"]:
    code=None; error=None
    try:
        with opener.open(url, timeout=c["timeout"]) as r: code=r.status
    except urllib.error.HTTPError as e: code=e.code
    except Exception as e: error=type(e).__name__
    denied=code in c["proxy_denied_statuses"] or error is not None
    out["proxy_denied"].append({"url":url,"denied":denied,"status":code,"error":error}); ok &= denied
for name in c["dns_names"]:
    try: socket.getaddrinfo(name,443); denied=False
    except socket.gaierror: denied=True
    out["dns_denied"].append({"name":name,"denied":denied}); ok &= denied
for row in c["direct_targets"]:
    try:
        s=socket.create_connection((row["host"],row["port"]),timeout=c["timeout"]); s.close(); denied=False
    except OSError: denied=True
    out["direct_denied"].append({**row,"denied":denied}); ok &= denied
out["ready"]=bool(ok); print(json.dumps(out,sort_keys=True)); sys.exit(0 if ok else 2)
'''


def run_agent_probes(binary: str, policy: Mapping[str, Any], provider: Mapping[str, Any], topology_result: Mapping[str, Any]) -> dict[str, Any]:
    topology = policy["topology"]
    image_ref = topology.get("agent_audit_image")
    expected_id = topology.get("agent_audit_image_id")
    if not isinstance(image_ref, str) or "@sha256:" not in image_ref or not isinstance(expected_id, str) or not expected_id.startswith("sha256:"):
        raise AuditError("agent audit image needs pinned manifest reference and local image ID")
    image = inspect_one(binary, "image", image_ref)
    if image.get("Id") != expected_id:
        raise AuditError("local agent audit image ID differs from preregistration")
    negative = policy["negative_probes"]
    timeout = policy.get("probe_timeout_seconds", 15)
    if not isinstance(timeout, int) or timeout < 1 or timeout > 60:
        raise AuditError("probe_timeout_seconds must be 1..60")
    denied_urls = negative.get("proxy_denied_urls")
    dns_names = negative.get("dns_denied_names")
    direct_targets = negative.get("direct_socket_denied")
    if not all(isinstance(value, list) and value for value in (denied_urls, dns_names, direct_targets)):
        raise AuditError("all negative probe lists must be nonempty")
    proxy_url = f"http://{topology_result['proxy_internal_ip']}:{topology['proxy_port']}"
    payload = {
        "proxy": proxy_url,
        "provider_url": provider["probe_url"],
        "provider_statuses": policy["provider_contract"].get("reachable_http_statuses", []),
        "proxy_denied_urls": denied_urls,
        "proxy_denied_statuses": [403],
        "dns_names": dns_names,
        "direct_targets": direct_targets,
        "timeout": timeout,
    }
    command = [
        binary, "run", "--rm", "--network", topology["agent_internal_network"],
        "--dns", "127.0.0.1", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--pids-limit", "64",
        "--memory", "128m", image_ref, "python3", "-c", PROBE_PROGRAM,
        json.dumps(payload, separators=(",", ":")),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout * (3 + len(denied_urls) + len(dns_names) + len(direct_targets)))
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AuditError(f"agent network probe returned invalid JSON; stderr={proc.stderr[-300:]!r}") from exc
    if proc.returncode != 0 or not isinstance(report, dict) or report.get("ready") is not True:
        raise AuditError(f"agent network probes failed: {report!r}")
    return {"image_id": image.get("Id"), "container_networks": [topology["agent_internal_network"]], "dns_server": "127.0.0.1", "mounts": [], "probes": report}


def build_attestation(
    *, policy: Mapping[str, Any], provider: Mapping[str, Any],
    topology: Mapping[str, Any], probes: Mapping[str, Any], docker_version: str,
    checked_at: str, host: str,
) -> dict[str, Any]:
    probe_rows = probes["probes"]
    direct_denied = all(row.get("denied") is True for row in probe_rows.get("direct_denied", []))
    dns_denied = all(row.get("denied") is True for row in probe_rows.get("dns_denied", []))
    proxy_denied = all(row.get("denied") is True for row in probe_rows.get("proxy_denied", []))
    provider_ok = probe_rows.get("provider", {}).get("reachable") is True
    proxy_endpoint = f"http://{topology['proxy_internal_ip']}:{policy['topology']['proxy_port']}"
    document: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "provider_proxy": proxy_endpoint,
        "docker_network": policy["topology"]["agent_internal_network"],
        "allowed_provider_hosts": [row["host"] for row in provider["provider_hosts"]],
        "enforcement": "docker-internal-network-plus-allowlist-proxy",
        "shell_public_egress_probe": {
            "passed": direct_denied and dns_denied and proxy_denied,
            "direct_public_egress_denied": direct_denied,
            "external_dns_denied": dns_denied,
            "github_via_proxy_denied": proxy_denied,
        },
        "provider_connectivity_probe": {
            "passed": provider_ok,
            "through_allowlist_proxy": provider_ok,
            "http_status": probe_rows.get("provider", {}).get("status"),
        },
        "audit_evidence": {
            "checked_at": checked_at,
            "host": host,
            "docker_server_version": docker_version,
            "provider_source_sha256": provider["provider_source_sha256"],
            "proxy_policy_sha256": provider["proxy_policy_sha256"],
            "topology": {key: value for key, value in topology.items() if key != "proxy_internal_ip"},
            "agent_probe_image_id": probes["image_id"],
            "agent_probe_networks": probes["container_networks"],
            "agent_probe_mounts": probes["mounts"],
            "negative_probes": {
                "proxy": probe_rows.get("proxy_denied", []),
                "dns": probe_rows.get("dns_denied", []),
                "direct": probe_rows.get("direct_denied", []),
            },
        },
        "failures": [],
    }
    document["ready"] = (
        document["shell_public_egress_probe"]["passed"]
        and document["provider_connectivity_probe"]["passed"]
    )
    document["audit_evidence"]["proof_sha256"] = digest(canonical_json(document["audit_evidence"]))
    return document


def run_audit(config_path: Path) -> dict[str, Any]:
    checked_at = dt.datetime.now(UTC).isoformat().replace("+00:00", "Z")
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "provider_proxy": None,
        "docker_network": None,
        "allowed_provider_hosts": [],
        "enforcement": "docker-internal-network-plus-allowlist-proxy",
        "shell_public_egress_probe": {"passed": False, "direct_public_egress_denied": False},
        "provider_connectivity_probe": {"passed": False, "through_allowlist_proxy": False},
        "audit_evidence": {"checked_at": checked_at, "host": socket.gethostname()},
        "ready": False,
        "failures": [],
    }
    try:
        policy = load_policy(config_path)
        binary = policy["docker"].get("binary", "docker")
        if not isinstance(binary, str) or shutil.which(binary) is None:
            raise AuditError("Docker binary is unavailable")
        version_proc = subprocess.run([binary, "version", "--format", "{{json .Server.Version}}"], capture_output=True, text=True, errors="replace", timeout=20)
        if version_proc.returncode != 0:
            raise AuditError("Docker daemon/version probe failed")
        observed = parse_version(version_proc.stdout)
        minimum = parse_version(str(policy["docker"].get("minimum_version", "")))
        if observed is None or minimum is None or not version_at_least(observed, minimum):
            raise AuditError("Docker server version is below preregistration")
        provider = validate_provider_contract(config_path, policy)
        topology_result = validate_topology(binary, policy, provider)
        probes = run_agent_probes(binary, policy, provider, topology_result)
        report = build_attestation(
            policy=policy, provider=provider, topology=topology_result, probes=probes,
            docker_version=".".join(map(str, observed)), checked_at=checked_at,
            host=socket.gethostname(),
        )
    except (AuditError, OSError, subprocess.TimeoutExpired) as exc:
        report["failures"].append({"code": "network_isolation", "message": f"{type(exc).__name__}: {exc}"})
    evidence = report.setdefault("audit_evidence", {})
    evidence["proof_sha256"] = digest(canonical_json(evidence))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_audit(args.config.resolve())
    if args.report:
        target = args.report.resolve()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except OSError as exc:
            report["ready"] = False
            report["failures"].append({"code": "report_write", "message": str(exc)})
            evidence = report.setdefault("audit_evidence", {})
            evidence.pop("proof_sha256", None)
            evidence["proof_sha256"] = digest(canonical_json(evidence))
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    sys.exit(main())
