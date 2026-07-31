from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench.agent_coding.v3 import audit_network_isolation as audit
from bench.agent_coding.v3 import runner


class ProviderContractTests(unittest.TestCase):
    def _policy(self, source_hash: str, policy_hash: str) -> dict[str, object]:
        return {
            "provider_contract": {
                "provider_source": "provider.js",
                "provider_source_sha256": source_hash,
                "allow_connect": [{"host": "api.minimax.io", "port": 443}],
                "proxy_policy_sha256": policy_hash,
                "probe_url": "https://api.minimax.io/anthropic",
            }
        }

    def test_allowlist_must_equal_frozen_provider_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = b'const baseUrl = "https://api.minimax.io/anthropic";\n'
            (root / "provider.js").write_bytes(source)
            allow = {"allow_connect": [{"host": "api.minimax.io", "port": 443}]}
            result = audit.validate_provider_contract(
                root / "network.json",
                self._policy(hashlib.sha256(source).hexdigest(), hashlib.sha256(audit.canonical_json(allow)).hexdigest()),
            )
            self.assertEqual(result["provider_hosts"], [{"host": "api.minimax.io", "port": 443}])

    def test_extra_non_provider_host_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = b'const baseUrl = "https://api.minimax.io/anthropic";\n'
            (root / "provider.js").write_bytes(source)
            policy = self._policy(hashlib.sha256(source).hexdigest(), "a" * 64)
            policy["provider_contract"]["allow_connect"].append({"host": "github.com", "port": 443})
            with self.assertRaises(audit.AuditError):
                audit.validate_provider_contract(root / "network.json", policy)


class TopologyTests(unittest.TestCase):
    def _policy(self) -> dict[str, object]:
        return {
            "topology": {
                "agent_internal_network": "internal",
                "proxy_egress_network": "egress",
                "proxy_container": "proxy",
                "proxy_policy_path": "/etc/greppy-proxy/policy.json",
                "proxy_image_id": "sha256:" + "1" * 64,
            },
            "mount_isolation": {
                "forbidden_root_env": ["GREPPY_BENCH_NAS_ROOT"],
                "forbidden_nvme_relative_paths": ["mirrors"],
                "proxy_allowed_mounts": [],
            },
        }

    def test_proxy_must_be_exactly_dual_homed_and_mount_free(self) -> None:
        policy = self._policy()
        baked = audit.canonical_json({"allow_connect": [{"host": "api.minimax.io", "port": 443}]})
        policy_hash = hashlib.sha256(baked).hexdigest()
        objects = {
            ("network", "internal"): {"Id": "n1", "Internal": True},
            ("network", "egress"): {"Id": "n2", "Internal": False},
            ("container", "proxy"): {
                "Id": "c1",
                "Image": "sha256:" + "1" * 64,
                "State": {"Running": True},
                "NetworkSettings": {"Networks": {"internal": {"IPAddress": "172.30.0.2"}, "egress": {"IPAddress": "172.31.0.2"}}},
                "Config": {"Labels": {"dev.greppy.v3.role": "allowlist-connect-proxy", "dev.greppy.v3.proxy-policy-sha256": policy_hash}},
                "Mounts": [],
            },
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ,
            {"GREPPY_BENCH_NAS_ROOT": str(Path(temporary) / "nas"), "GREPPY_BENCH_NVME_ROOT": str(Path(temporary) / "nvme")},
            clear=False,
        ), mock.patch.object(audit, "inspect_one", side_effect=lambda binary, kind, name: objects[(kind, name)]), mock.patch.object(
            audit.subprocess, "run", return_value=mock.Mock(returncode=0, stdout=baked)
        ):
            result = audit.validate_topology("docker", policy, {"proxy_policy_sha256": policy_hash})
        self.assertEqual(result["proxy"]["networks"], ["egress", "internal"])
        self.assertEqual(result["proxy"]["mount_count"], 0)

    def test_third_proxy_network_is_rejected(self) -> None:
        policy = self._policy()
        proxy = {
            "Id": "c1", "Image": "sha256:" + "1" * 64, "State": {"Running": True},
            "NetworkSettings": {"Networks": {"internal": {}, "egress": {}, "bridge": {}}},
            "Config": {"Labels": {}}, "Mounts": [],
        }
        objects = {
            ("network", "internal"): {"Internal": True},
            ("network", "egress"): {"Internal": False},
            ("container", "proxy"): proxy,
        }
        with mock.patch.object(audit, "inspect_one", side_effect=lambda binary, kind, name: objects[(kind, name)]):
            with self.assertRaises(audit.AuditError):
                audit.validate_topology("docker", policy, {"proxy_policy_sha256": "a" * 64})


class ProbeCommandTests(unittest.TestCase):
    def test_probe_container_has_only_internal_network_and_no_secret_env(self) -> None:
        policy = {
            "topology": {
                "agent_internal_network": "internal",
                "proxy_port": 3128,
                "agent_audit_image": "python:3.12-alpine@sha256:" + "2" * 64,
                "agent_audit_image_id": "sha256:" + "3" * 64,
            },
            "negative_probes": {
                "proxy_denied_urls": ["https://github.com/"],
                "dns_denied_names": ["github.com"],
                "direct_socket_denied": [{"host": "1.1.1.1", "port": 443}],
            },
            "provider_contract": {"reachable_http_statuses": [401, 404]},
            "probe_timeout_seconds": 2,
        }
        provider = {"probe_url": "https://api.minimax.io/anthropic"}
        probe_report = {
            "ready": True,
            "provider": {"reachable": True, "status": 401},
            "proxy_denied": [{"url": "https://github.com/", "denied": True, "status": 403}],
            "dns_denied": [{"name": "github.com", "denied": True}],
            "direct_denied": [{"host": "1.1.1.1", "port": 443, "denied": True}],
        }
        completed = mock.Mock(returncode=0, stdout=json.dumps(probe_report), stderr="")
        with mock.patch.object(audit, "inspect_one", return_value={"Id": "sha256:" + "3" * 64}), mock.patch.object(audit.subprocess, "run", return_value=completed) as run:
            result = audit.run_agent_probes("docker", policy, provider, {"proxy_internal_ip": "172.30.0.2"})
        command = run.call_args.args[0]
        self.assertIn("--network", command)
        self.assertEqual(command[command.index("--network") + 1], "internal")
        self.assertEqual(command[command.index("--dns") + 1], "127.0.0.1")
        self.assertNotIn("--env", command)
        self.assertNotIn("MINIMAX_API_KEY", " ".join(command))
        self.assertEqual(result["mounts"], [])


class ReportTests(unittest.TestCase):
    def test_unsealed_template_fails_with_hashed_machine_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            path.write_text(json.dumps({"schema_version": audit.POLICY_SCHEMA, "status": "template-not-sealed"}), encoding="utf-8")
            report = audit.run_audit(path)
            self.assertFalse(report["ready"])
            proof = report["audit_evidence"].pop("proof_sha256")
            self.assertEqual(proof, hashlib.sha256(audit.canonical_json(report["audit_evidence"])).hexdigest())

    def test_real_audit_output_is_the_runner_attestation_schema(self) -> None:
        policy = {
            "topology": {"agent_internal_network": "internal", "proxy_port": 3128},
        }
        provider = {
            "provider_hosts": [{"host": "api.minimax.io", "port": 443}],
            "provider_source_sha256": "a" * 64,
            "proxy_policy_sha256": "b" * 64,
        }
        topology = {
            "proxy_internal_ip": "172.30.0.2",
            "internal_network": {"name": "internal", "internal": True},
            "egress_network": {"name": "egress", "internal": False},
            "proxy": {"networks": ["egress", "internal"], "mount_count": 0},
        }
        probes = {
            "image_id": "sha256:" + "c" * 64,
            "container_networks": ["internal"],
            "mounts": [],
            "probes": {
                "provider": {"reachable": True, "status": 401},
                "proxy_denied": [{"url": "https://github.com/", "denied": True, "status": 403}],
                "dns_denied": [{"name": "github.com", "denied": True}],
                "direct_denied": [{"host": "1.1.1.1", "port": 443, "denied": True}],
            },
        }
        document = audit.build_attestation(
            policy=policy, provider=provider, topology=topology, probes=probes,
            docker_version="27.2.0", checked_at="2026-07-31T00:00:00Z", host="gpu3",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attestation.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = runner.load_network_attestation(path, "http://172.30.0.2:3128", "internal")
        self.assertEqual(loaded.allowed_provider_hosts, ("api.minimax.io",))


if __name__ == "__main__":
    unittest.main()
