from __future__ import annotations

import hashlib
import io
import json
import pathlib
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

try:
    from . import runner
except ImportError:
    import runner


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RunnerFixture(unittest.TestCase):
    TASK_ID = "task_aaaaaaaaaaaaaaaaaaaaaaaaaa"
    PARENT = "1" * 40
    GOLD = "2" * 40

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="v3-runner-test-")
        self.root = pathlib.Path(self.temporary.name)
        self.public = self.root / "release" / "public"
        self.sealed = self.root / "release" / "sealed"
        (self.public / "snapshots").mkdir(parents=True)
        (self.sealed / "patches").mkdir(parents=True)
        self.snapshot = self.public / "snapshots" / f"{self.TASK_ID}.tar"
        with tarfile.open(self.snapshot, "w") as archive:
            for name, content in {
                "value.txt": b"old\n",
                "test_existing.py": b"import pathlib\nassert pathlib.Path('value.txt').read_text() == 'new\\n'\n",
            }.items():
                info = tarfile.TarInfo(name)
                info.mode = 0o644
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        self.test_patch = (
            "diff --git a/test_hidden.py b/test_hidden.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/test_hidden.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+import pathlib\n"
            "+assert pathlib.Path('value.txt').read_text() == 'new\\n'\n"
        ).encode()
        (self.sealed / "patches" / f"{self.TASK_ID}.test.patch").write_bytes(self.test_patch)
        public_doc = {
            "schema_version": runner.SCHEMA_VERSION,
            "freeze": {"id": "freeze-test"},
            "tasks": [{
                "id": self.TASK_ID,
                "user_task": "Change the value from old to new.",
                "workspace": {
                    "snapshot": f"snapshots/{self.TASK_ID}.tar",
                    "snapshot_sha256": sha(self.snapshot.read_bytes()),
                },
            }],
        }
        evaluation = {
            "test_command": [sys.executable],
            "setup_commands": [],
            "post_patch_commands": [[
                sys.executable, "-c",
                "import pathlib; assert pathlib.Path('test_hidden.py').exists(); "
                "assert pathlib.Path('value.txt').read_text() == 'new\\n'",
            ]],
            "timeout_seconds": 10,
        }
        sealed_doc = {
            "schema_version": runner.SEALED_SCHEMA,
            "freeze_id": "freeze-test",
            "tasks": [{
                "id": self.TASK_ID,
                "pr_number": 98765,
                "parent_commit": self.PARENT,
                "solution_commit": self.GOLD,
                "artifacts": {"test_patch": f"patches/{self.TASK_ID}.test.patch"},
                "hashes": {
                    "test_patch_sha256": sha(self.test_patch),
                    "evaluation_sha256": sha(runner.compact_canonical_json(evaluation)),
                },
                "evaluation": evaluation,
                "validation_evidence": {
                    "fail_to_pass": ["test_hidden.py"],
                    "pass_to_pass": ["test_existing.py"],
                },
            }],
        }
        (self.public / "taskbank.json").write_text(json.dumps(public_doc), encoding="utf-8")
        (self.sealed / "manifest.json").write_text(json.dumps(sealed_doc), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def pair(self):
        return runner.load_release(self.public, self.sealed)[0]


class IsolationAndGradingTests(RunnerFixture):
    def test_agent_never_sees_sealed_tests_or_identifiers_and_grading_applies_them_afterward(self) -> None:
        public_task, sealed_task = self.pair()
        calls: list[runner.AgentRequest] = []

        def fake_agent(request: runner.AgentRequest) -> runner.AgentOutcome:
            calls.append(request)
            self.assertFalse((request.workspace / "test_hidden.py").exists())
            exposed = "\n".join([
                str(request.workspace), str(request.store), str(request.raw_dir),
                request.system_prompt, request.user_prompt,
                *request.environment.keys(), *request.environment.values(),
            ])
            for forbidden in (self.TASK_ID, self.PARENT, self.GOLD, "98765"):
                self.assertNotIn(forbidden, exposed)
            history = runner.checked(
                ["git", "log", "--all", "--oneline"], cwd=request.workspace, timeout_seconds=10
            ).stdout.decode().splitlines()
            self.assertEqual(len(history), 1)
            self.assertEqual(
                runner.checked(["git", "remote"], cwd=request.workspace, timeout_seconds=10).stdout,
                b"",
            )
            (request.workspace / "value.txt").write_text("new\n", encoding="utf-8")
            return runner.AgentOutcome(0)

        trusted_calls: list[tuple[list[str], bool]] = []

        def trusted_command(argv, *, cwd, timeout_seconds):
            trusted_calls.append((list(argv), (cwd / "test_hidden.py").exists()))
            return runner.run_command(argv, cwd=cwd, timeout_seconds=timeout_seconds)

        rows = runner.execute_pair(
            public_task=public_task, sealed_task=sealed_task,
            public_dir=self.public, sealed_dir=self.sealed,
            slot_dir=self.root / "work" / "slot-0001",
            agents_md="SHIPPED MANUAL", executor=fake_agent,
            command_executor=trusted_command,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].tools, calls[1].tools)
        for row in rows:
            self.assertTrue(row["grading"]["correctness"])
            self.assertTrue(row["grading"]["agent_diff_applied_before_test_patch"])
            self.assertTrue(row["grading"]["test_patch_applied_after_agent"])
            self.assertTrue(row["grading"]["post_patch_commands_executed"])
            patch = (self.root / "work" / "slot-0001" / row["arm"] / "raw" / "agent.patch").read_text()
            self.assertNotIn("test_hidden.py", patch)
        self.assertEqual(len(trusted_calls), 6)
        self.assertTrue(all(hidden_exists for _, hidden_exists in trusted_calls))
        for offset in (0, 3):
            self.assertEqual(trusted_calls[offset][0][:2], [sys.executable, "-c"])

    def test_hidden_test_is_not_applied_when_agent_phase_fails_before_grading(self) -> None:
        public_task, sealed_task = self.pair()

        def failing_agent(request: runner.AgentRequest) -> runner.AgentOutcome:
            self.assertFalse((request.workspace / "test_hidden.py").exists())
            raise RuntimeError("agent stopped")

        rows = runner.execute_pair(
            public_task=public_task, sealed_task=sealed_task,
            public_dir=self.public, sealed_dir=self.sealed,
            slot_dir=self.root / "failed" / "slot-0001",
            agents_md="manual", executor=failing_agent,
        )
        self.assertTrue(all(not row["valid"] for row in rows))
        self.assertTrue(all(row["agent"]["metrics"]["provider_cost_usd"] is None for row in rows))
        self.assertFalse((self.root / "failed" / "slot-0001" / "control" / "agent-workspace" / "test_hidden.py").exists())

    def test_prompts_use_runtime_manual_and_identical_tool_palettes(self) -> None:
        manual = "runtime shipped AGENTS.md"
        self.assertNotIn(manual, runner.system_prompt("control", manual))
        self.assertTrue(runner.system_prompt("treatment", manual).endswith(manual))
        self.assertEqual(len(set(runner.ARM_TOOLS.values())), 1)

    def test_arm_order_is_exactly_balanced(self) -> None:
        for count, expected in ((41, (21, 20)), (144, (72, 72))):
            task_ids = [f"task-{index:03d}" for index in range(count)]
            orders = runner.balanced_arm_orders(task_ids)
            control_first = sum(order[0] == "control" for order in orders.values())
            self.assertEqual((control_first, count - control_first), expected)

    def test_test_patch_conflict_is_a_task_failure_not_a_runner_crash(self) -> None:
        _, sealed_task = self.pair()
        workspace = self.root / "conflict-grade"
        conflicting_agent_diff = (
            "diff --git a/test_hidden.py b/test_hidden.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/test_hidden.py\n"
            "@@ -0,0 +1 @@\n"
            "+agent-owned\n"
        ).encode()
        grade = runner.grade_agent_diff(
            snapshot=self.snapshot, sealed_dir=self.sealed, sealed_task=sealed_task,
            agent_diff=conflicting_agent_diff, workspace=workspace,
        )
        self.assertFalse(grade["correctness"])
        self.assertFalse(grade["test_patch_applied_after_agent"])
        self.assertIn("patch_application_error", grade)

    def test_second_arm_failure_preserves_first_arm_paid_metrics(self) -> None:
        public_task, sealed_task = self.pair()
        calls = 0

        def agent(request: runner.AgentRequest) -> runner.AgentOutcome:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("second arm crashed")
            (request.workspace / "value.txt").write_text("new\n", encoding="utf-8")
            return runner.AgentOutcome(0, metrics={"provider_cost_usd": 1.25})

        rows = runner.execute_pair(
            public_task=public_task, sealed_task=sealed_task,
            public_dir=self.public, sealed_dir=self.sealed,
            slot_dir=self.root / "partial" / "slot-0001", agents_md="manual",
            executor=agent,
        )
        self.assertEqual(rows[0]["agent"]["metrics"]["provider_cost_usd"], 1.25)
        self.assertIsNone(rows[1]["agent"]["metrics"]["provider_cost_usd"])
        self.assertFalse(rows[1]["valid"])


class SnapshotSafetyTests(RunnerFixture):
    def test_snapshot_rejects_git_metadata_and_traversal(self) -> None:
        for name in (".git/config", "../gold.patch"):
            with self.subTest(name=name):
                archive_path = self.root / (name.replace("/", "_") + ".tar")
                with tarfile.open(archive_path, "w") as archive:
                    payload = b"secret"
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                with self.assertRaises(runner.RunnerError):
                    runner.import_parent_snapshot(archive_path, self.root / ("out-" + archive_path.stem), 10)

    def test_release_hash_mismatch_blocks_before_execution(self) -> None:
        self.snapshot.write_bytes(self.snapshot.read_bytes() + b"tampered")
        with self.assertRaisesRegex(runner.RunnerError, "snapshot hash mismatch"):
            runner.load_release(self.public, self.sealed)

    def test_missing_or_hash_mismatched_post_patch_spec_blocks_loading(self) -> None:
        manifest_path = self.sealed / "manifest.json"
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing = json.loads(json.dumps(original))
        missing["tasks"][0]["evaluation"].pop("post_patch_commands")
        missing["tasks"][0]["hashes"]["evaluation_sha256"] = sha(
            runner.compact_canonical_json(missing["tasks"][0]["evaluation"])
        )
        manifest_path.write_text(json.dumps(missing), encoding="utf-8")
        with self.assertRaisesRegex(runner.RunnerError, "post_patch_commands"):
            runner.load_release(self.public, self.sealed)

        mismatched = json.loads(json.dumps(original))
        mismatched["tasks"][0]["evaluation"]["post_patch_commands"].append(
            [sys.executable, "-c", "pass"]
        )
        manifest_path.write_text(json.dumps(mismatched), encoding="utf-8")
        with self.assertRaisesRegex(runner.RunnerError, "evaluation spec hash mismatch"):
            runner.load_release(self.public, self.sealed)


class NetworkIsolationTests(unittest.TestCase):
    def test_attestation_requires_denied_shell_egress_and_working_provider_proxy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v3-network-test-") as temporary:
            path = pathlib.Path(temporary) / "attestation.json"
            document = {
                "schema_version": "greppy.provider-only-egress.v1",
                "provider_proxy": "http://172.28.0.2:3128",
                "docker_network": "greppy-agent-internal",
                "allowed_provider_hosts": ["api.minimax.io"],
                "enforcement": "docker-internal-network-plus-allowlist-proxy",
                "shell_public_egress_probe": {
                    "passed": True,
                    "direct_public_egress_denied": True,
                },
                "provider_connectivity_probe": {
                    "passed": True,
                    "through_allowlist_proxy": True,
                },
                "audit_evidence": {
                    "agent_probe_image_id": "sha256:" + "b" * 64,
                    "topology": {
                        "internal_network": {"name": "greppy-agent-internal", "id": "net-in", "internal": True},
                        "egress_network": {"name": "greppy-proxy-egress", "id": "net-out", "internal": False},
                        "proxy": {
                            "name": "greppy-proxy", "id": "proxy-id",
                            "image_id": "sha256:" + "e" * 64,
                            "networks": ["greppy-agent-internal", "greppy-proxy-egress"],
                            "mount_count": 0,
                        },
                    },
                },
            }
            document["audit_evidence"]["proof_sha256"] = runner.sha256(
                runner.compact_canonical_json(document["audit_evidence"])
            )
            path.write_text(json.dumps(document), encoding="utf-8")
            attestation = runner.load_network_attestation(
                path, "http://172.28.0.2:3128", "greppy-agent-internal"
            )
            self.assertEqual(attestation.allowed_provider_hosts, ("api.minimax.io",))
            self.assertEqual(attestation.agent_probe_image_id, "sha256:" + "b" * 64)
            runner.verify_network_attestation_image(attestation, "sha256:" + "b" * 64)
            with self.assertRaisesRegex(runner.RunnerError, "different agent image"):
                runner.verify_network_attestation_image(attestation, "sha256:" + "d" * 64)
            with self.assertRaisesRegex(runner.RunnerError, "internal IP"):
                runner.load_network_attestation(
                    path, "http://provider-proxy:3128", "greppy-agent-internal"
                )
            document["audit_evidence"]["topology"]["proxy"]["id"] = "tampered"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(runner.RunnerError, "does not prove"):
                runner.load_network_attestation(
                    path, "http://172.28.0.2:3128", "greppy-agent-internal"
                )
            document["audit_evidence"]["topology"]["proxy"]["id"] = "proxy-id"
            document["shell_public_egress_probe"]["direct_public_egress_denied"] = False
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(runner.RunnerError, "denied shell egress"):
                runner.load_network_attestation(
                    path, "http://172.28.0.2:3128", "greppy-agent-internal"
                )

    def test_attestation_proof_and_live_docker_ids_are_revalidated(self) -> None:
        topology = {
            "internal_network": {"name": "internal", "id": "net-in", "internal": True},
            "egress_network": {"name": "egress", "id": "net-out", "internal": False},
            "proxy": {
                "name": "proxy", "id": "proxy-id", "image_id": "sha256:" + "e" * 64,
                "networks": ["egress", "internal"], "mount_count": 0,
            },
        }
        attestation = runner.NetworkAttestation(
            "http://172.28.0.2:3128", "internal", ("api.minimax.io",),
            "sha256:" + "b" * 64, "c" * 64, "a" * 64, topology,
        )
        live = {
            ("network", "internal"): {"Id": "net-in", "Internal": True},
            ("network", "egress"): {"Id": "net-out", "Internal": False},
            ("container", "proxy"): {
                "Id": "proxy-id", "Image": "sha256:" + "e" * 64,
                "State": {"Running": True}, "Mounts": [],
                "NetworkSettings": {"Networks": {
                    "egress": {"IPAddress": "10.0.0.2"},
                    "internal": {"IPAddress": "172.28.0.2"},
                }},
            },
        }
        with mock.patch.object(
            runner, "_docker_inspect_one", side_effect=lambda _docker, kind, name: live[(kind, name)]
        ):
            runner.verify_live_network_attestation(pathlib.Path("docker"), attestation)
            live[("container", "proxy")]["Id"] = "replaced-proxy"
            with self.assertRaisesRegex(runner.RunnerError, "differ"):
                runner.verify_live_network_attestation(pathlib.Path("docker"), attestation)

    def test_ephemeral_provider_key_is_mounted_not_exposed_in_docker_arguments(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v3-secret-test-") as temporary:
            root = pathlib.Path(temporary)
            for name in ("docker", "greppy", "provider.js"):
                (root / name).write_text("x", encoding="utf-8")
            secret = b"scoped-ephemeral-secret-value"
            key_file = root / "provider.key"
            key_file.write_bytes(secret)
            key_file.chmod(0o600)
            workspace, store, raw = root / "workspace", root / "store", root / "raw"
            for path in (workspace, store, raw):
                path.mkdir()
            network = runner.NetworkAttestation(
                "http://172.28.0.2:3128", "internal", ("api.minimax.io",),
                "sha256:" + "b" * 64, "c" * 64, "a" * 64,
            )
            executor = runner.DockerPiExecutor(
                docker=root / "docker", image="agent@sha256:" + "b" * 64,
                network=network, pi_command="pi", greppy_bin=root / "greppy",
                provider_extension=root / "provider.js", provider_key_file=key_file,
                provider="minimax", model="model",
                shipped_commands=frozenset({"read"}), shipped_edits=frozenset({"replace"}),
            )
            request = runner.AgentRequest(
                arm="control", workspace=workspace, store=store, raw_dir=raw,
                system_prompt="system", user_prompt="task", tools="bash,read,edit,write",
                environment={}, timeout_seconds=10,
            )
            result = runner.CommandResult(
                ("docker",), 0, b"output " + secret, b"", 0.1, False
            )
            with mock.patch.object(runner, "run_command", return_value=result) as invoked:
                outcome = executor(request)
            docker_argv = invoked.call_args.args[0]
            self.assertNotIn(secret.decode(), "\n".join(docker_argv))
            self.assertIn(str(key_file), "\n".join(docker_argv))
            self.assertIn("--dns", docker_argv)
            self.assertEqual(docker_argv[docker_argv.index("--dns") + 1], "127.0.0.1")
            self.assertNotIn(secret, outcome.stdout)
            self.assertIn(b"<redacted>", outcome.stdout)


class SummaryTests(unittest.TestCase):
    def test_zero_denominator_rate_is_na_and_cannot_pass_a_gate(self) -> None:
        metric = runner.rate_metric(0, 0)
        self.assertEqual(metric["rate"], "N/A")
        self.assertFalse(metric["gate_eligible"])
        self.assertFalse(runner.rate_gate_passes(metric, 0.0))
        summary = runner.summarize_results([], "manual")
        self.assertEqual(
            summary["greppy_task_adoption_diagnostic_only"]["rate"], "N/A"
        )
        self.assertFalse(summary["correctness_noninferiority"]["passes"])

    def test_runtime_command_vocabulary_drives_greppy_adoption_metrics(self) -> None:
        manual = "READ:\n  read S    source\n\nEDIT:\n  replace S N    edit\n"
        commands, edits = runner.commands_from_agents_md(manual)
        event = {
            "type": "turn_end",
            "message": {
                "usage": {},
                "content": [{
                    "type": "toolCall", "name": "bash",
                    "arguments": {"command": "greppy --root . read thing && greppy --root . replace thing new"},
                }],
            },
            "toolResults": [],
        }
        metrics = runner.parse_pi_metrics((json.dumps(event) + "\n").encode(), commands, edits)
        self.assertEqual(metrics["greppy_calls"], 2)
        self.assertEqual(metrics["greppy_edit_calls"], 1)
        self.assertEqual(
            metrics["transactionality_observation"],
            "unobservable_without_per_tool_interception",
        )
        self.assertNotIn("partial_state_incidents", metrics)

    def test_log_words_do_not_fake_transactionality_observation(self) -> None:
        event = {
            "type": "turn_end", "message": {"usage": {}, "content": []},
            "toolResults": [{"text": "partial state partially applied"}],
        }
        metrics = runner.parse_pi_metrics((json.dumps(event) + "\n").encode())
        self.assertNotIn("partial_state_incidents", metrics)

    def test_source_opens_count_builtin_shell_and_greppy_source_output(self) -> None:
        event = {
            "type": "turn_end",
            "message": {"usage": {}, "content": [
                {"type": "toolCall", "name": "read", "arguments": {"path": "a.py"}},
                {"type": "toolCall", "name": "bash", "arguments": {
                    "command": "cat b.py; sed -n '1,20p' c.py; greppy read d.py && greppy who-calls x --code",
                }},
            ]},
        }
        metrics = runner.parse_pi_metrics((json.dumps(event) + "\n").encode())
        self.assertEqual(metrics["source_open_by_kind"], {
            "builtin_read": 1, "shell_reader": 2, "greppy_source": 2,
        })
        self.assertEqual(metrics["source_open_events"], 5)

    def test_primary_cost_is_intention_to_treat_and_includes_failed_tasks(self) -> None:
        def arm(name: str, cost: float, correct: bool):
            return {
                "arm": name,
                "valid": True,
                "agent": {"metrics": {
                    "provider_cost_usd": cost, "input_tokens": 100,
                    "greppy_calls": 1 if name == "treatment" else 0,
                }, "wall_seconds": 1},
                "grading": {"correctness": correct},
            }

        rows = [
            {
                "strata": {"repository": "r", "language": "python", "task_class": "bug"},
                "arms": [arm("control", 1.0, True), arm("treatment", 0.5, True)],
            },
            {
                "strata": {"repository": "r", "language": "python", "task_class": "bug"},
                "arms": [arm("control", 2.0, False), arm("treatment", 4.0, False)],
            },
        ]
        summary = runner.summarize_results(rows, "manual")
        self.assertEqual(summary["primary_cost_population"], "all_tasks_all_provider_attempts_intention_to_treat")
        self.assertEqual(summary["arms"]["control"]["total_provider_cost_usd_all_tasks_all_attempts"], 3.0)
        self.assertEqual(summary["arms"]["treatment"]["total_provider_cost_usd_all_tasks_all_attempts"], 4.5)
        self.assertEqual(summary["arms"]["treatment"]["cost_per_solve_usd"], 4.5)
        self.assertEqual(summary["solved_pair_cost_ratio_descriptive_only"]["n"], 1)
        self.assertEqual(summary["paired_efficiency_intervals"]["gross_input_tokens"]["n"], 2)
        self.assertEqual(summary["paired_efficiency_intervals"]["agent_wall_seconds"]["n"], 2)
        self.assertFalse(summary["transactionality_observation"]["observable"])
        self.assertFalse(summary["transactionality_observation"]["is_release_gate"])


class OperationalEvidenceTests(unittest.TestCase):
    def test_failed_full_release_gate_exits_nonzero_but_smoke_subset_does_not(self) -> None:
        self.assertEqual(runner.release_process_exit_code(True, False), 3)
        self.assertEqual(runner.release_process_exit_code(True, None), 3)
        self.assertEqual(runner.release_process_exit_code(True, True), 0)
        self.assertEqual(runner.release_process_exit_code(False, None), 0)

    def test_unresolved_provider_credential_boundary_blocks_full_but_not_smoke(self) -> None:
        runner.enforce_credential_boundary(False)
        with self.assertRaisesRegex(runner.RunnerError, "full run blocked"):
            runner.enforce_credential_boundary(True)

    def test_signed_preflight_and_exact_three_pair_smoke_bind_current_runtime(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v3-ops-evidence-") as temporary:
            root = pathlib.Path(temporary)
            report_path = root / "preflight.json"
            attestation_path = root / "preflight-attestation.json"
            preflight_signature = root / "preflight.sig"
            smoke_path = root / "smoke.json"
            smoke_signature = root / "smoke.sig"
            public_key = root / "ops.pem"
            openssl = root / "openssl"
            for path in (preflight_signature, smoke_signature, public_key, openssl):
                path.write_bytes(b"test")
            network = runner.NetworkAttestation(
                "http://172.28.0.2:3128", "internal", ("api.minimax.io",),
                "sha256:" + "b" * 64, "c" * 64, "d" * 64,
            )
            bindings = {"runner_source_sha256": "a" * 64, "model": {"provider": "p", "model": "m"}}
            report = {
                "schema_version": runner.PREFLIGHT_REPORT_SCHEMA, "ready": True,
                "failures": [], "checks": {"network": {
                    "ready": True, "audit_evidence": {"proof_sha256": network.proof_sha256},
                }},
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            attestation = {
                "schema_version": runner.PREFLIGHT_ATTESTATION_SCHEMA, "ready": True,
                "preflight_report_sha256": sha(report_path.read_bytes()),
                "runtime_bindings": bindings,
            }
            attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
            with mock.patch.object(runner, "verify_detached_signature"):
                preflight = runner.load_signed_preflight(
                    report_path=report_path, attestation_path=attestation_path,
                    signature_path=preflight_signature, public_key=public_key,
                    openssl=openssl, runtime_bindings=bindings, network=network,
                )
                smoke = {
                    "schema_version": runner.SMOKE_EVIDENCE_SCHEMA, "ready": True,
                    "paired_trajectory_count": 3, "arm_trace_count": 6,
                    "task_ids": ["task-a", "task-b", "task-c"],
                    "smoke_run_archive_sha256": "e" * 64,
                    "arm_trace_sha256": [f"{index:064x}" for index in range(6)],
                    "manual_review": {
                        "passed": True, "read_all_six_arm_traces": True,
                        "open_findings": [],
                    },
                    "runtime_bindings": bindings,
                    "preflight_attestation_sha256": preflight["attestation_sha256"],
                }
                smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
                runner.load_signed_smoke_evidence(
                    evidence_path=smoke_path, signature_path=smoke_signature,
                    public_key=public_key, openssl=openssl,
                    runtime_bindings=bindings,
                    preflight_attestation_sha256=preflight["attestation_sha256"],
                )
                smoke["paired_trajectory_count"] = 2
                smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
                with self.assertRaisesRegex(runner.RunnerError, "exactly three"):
                    runner.load_signed_smoke_evidence(
                        evidence_path=smoke_path, signature_path=smoke_signature,
                        public_key=public_key, openssl=openssl,
                        runtime_bindings=bindings,
                        preflight_attestation_sha256=preflight["attestation_sha256"],
                    )


if __name__ == "__main__":
    unittest.main()
