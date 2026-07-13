#!/usr/bin/env python3
"""Unit tests for the inference performance calibration verifier."""

from __future__ import annotations

import copy
import unittest

from bench.inference_performance import contract, verify


def _hash(character: str) -> str:
    return character * 64


def _hardware(platform: str) -> dict[str, object]:
    return {
        "platform": platform,
        "cpu": f"fixture-{platform}-cpu",
        "gpus": [] if platform.endswith("cpu") else [f"fixture-{platform}-gpu0"],
        "memory_bytes": 32 * 1024**3,
    }


def _device(platform: str) -> dict[str, object]:
    if platform.endswith("cpu"):
        return {"kind": "cpu", "id": "cpu", "gpu_count": 0, "visible_gpu_ids": []}
    return {
        "kind": platform,
        "id": "0",
        "gpu_count": 1,
        "visible_gpu_ids": ["0"],
    }


def _source_hash(engine: str) -> str:
    return _hash("a" if engine == "native" else "b")


def _binary_hash(engine: str, platform: str, model_family: str) -> str:
    value = f"{engine}:{platform}:{model_family}".encode("ascii")
    return contract.sha256_bytes(value)


def _model_hash(model_family: str) -> str:
    return _hash("c" if model_family == "qwen35_mtp" else "d")


def _tokenizer_hash(model_family: str) -> str:
    return _hash("e" if model_family == "qwen35_mtp" else "f")


def _raw_record(
    *,
    platform: str,
    engine: str,
    workload: str,
    sample_index: int,
    rate: float,
) -> dict[str, object]:
    model_family = "embeddinggemma" if workload == contract.EMBEDDING_ENCODER else "qwen35_mtp"
    if workload == contract.QWEN_PP512:
        input_ids = list(range(512))
        output_ids: list[int] = []
        output_limit = 0
        generation_path = "target_prefill"
    elif workload == contract.QWEN_TG128:
        input_ids = [17]
        output_ids = list(range(1000, 1128))
        output_limit = 128
        generation_path = "production_mtp" if engine == "native" else "target_greedy_reference"
    elif workload == contract.EMBEDDING_ENCODER:
        input_ids = list(range(128))
        output_ids = []
        output_limit = 0
        generation_path = "encoder"
    else:
        input_ids = list(range(120))
        output_ids = []
        output_limit = 64
        generation_path = "production_mtp"
    rate_tokens = len(output_ids) if workload == contract.QWEN_TG128 else len(input_ids)
    elapsed_ns = round(rate_tokens * 1_000_000_000 / rate)
    hardware = _hardware(platform)
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "run_id": "fixture-run",
        "platform": platform,
        "engine": engine,
        "model_family": model_family,
        "workload": workload,
        "semantics": contract.SEMANTICS[workload],
        "generation_path": generation_path,
        "case_id": f"{workload}-case",
        "sample_index": sample_index,
        "elapsed_ns": elapsed_ns,
        "latency_ms": elapsed_ns / 1_000_000.0,
        "tokens_per_second": rate_tokens * 1_000_000_000.0 / elapsed_ns,
        "input_tokens": len(input_ids),
        "output_tokens": len(output_ids),
        "output_limit": output_limit,
        "input_token_ids": input_ids,
        "output_token_ids": output_ids,
        "attention_mask": [1] * len(input_ids) if workload == contract.EMBEDDING_ENCODER else [],
        "input_token_ids_sha256": contract.token_ids_sha256(input_ids),
        "output_token_ids_sha256": contract.token_ids_sha256(output_ids) if output_ids else None,
        "attention_mask_sha256": (
            contract.token_ids_sha256([1] * len(input_ids))
            if workload == contract.EMBEDDING_ENCODER
            else None
        ),
        "threads": 4,
        "p_core_set": ["p0", "p1", "p2", "p3"],
        "device": _device(platform),
        "hardware": hardware,
        "hardware_sha256": contract.hardware_sha256(hardware),
        "binary_sha256": _binary_hash(engine, platform, model_family),
        "model_sha256": _model_hash(model_family),
        "tokenizer_sha256": _tokenizer_hash(model_family),
        "source_sha256": _source_hash(engine),
    }


def valid_records() -> list[dict[str, object]]:
    records = []
    native_rates = [108.0, 109.0, 110.0, 111.0, 112.0]
    llama_rates = [98.0, 99.0, 100.0, 101.0, 102.0]
    for platform in contract.PLATFORMS:
        for workload in contract.GATED_WORKLOADS:
            for engine, rates in (("native", native_rates), ("llama.cpp", llama_rates)):
                for sample_index, rate in enumerate(rates):
                    records.append(
                        _raw_record(
                            platform=platform,
                            engine=engine,
                            workload=workload,
                            sample_index=sample_index,
                            rate=rate,
                        )
                    )
        for sample_index, rate in enumerate(native_rates):
            records.append(
                _raw_record(
                    platform=platform,
                    engine="native",
                    workload=contract.GREPPY_BRIEF,
                    sample_index=sample_index,
                    rate=rate,
                )
            )
    return records


def _rewrite_rate(record: dict[str, object], rate: float) -> None:
    rate_tokens = (
        int(record["output_tokens"])
        if record["workload"] == contract.QWEN_TG128
        else int(record["input_tokens"])
    )
    elapsed_ns = round(rate_tokens * 1_000_000_000 / rate)
    record["elapsed_ns"] = elapsed_ns
    record["latency_ms"] = elapsed_ns / 1_000_000.0
    record["tokens_per_second"] = rate_tokens * 1_000_000_000.0 / elapsed_ns


class InferencePerformanceVerifierTests(unittest.TestCase):
    def test_valid_four_platform_result_passes(self) -> None:
        report = verify.verify_records(valid_records())
        self.assertEqual(report["status"], "pass")
        self.assertEqual(len(report["gates"]), 12)
        self.assertTrue(all(gate["median_speedup"] >= 1.05 for gate in report["gates"]))

    def test_missing_llama_baseline_fails(self) -> None:
        records = [
            record
            for record in valid_records()
            if not (
                record["platform"] == "apple_cpu"
                and record["workload"] == contract.QWEN_PP512
                and record["engine"] == "llama.cpp"
            )
        ]
        self.assert_fails(records, "missing mandatory engines ['llama.cpp']")

    def test_failed_pp511_claim_is_rejected(self) -> None:
        records = valid_records()
        for record in records:
            if record["workload"] == contract.QWEN_PP512:
                ids = list(record["input_token_ids"])[:-1]
                record["input_token_ids"] = ids
                record["input_tokens"] = len(ids)
                record["input_token_ids_sha256"] = contract.token_ids_sha256(ids)
                _rewrite_rate(record, float(record["tokens_per_second"]))
        self.assert_fails(records, "PP512 must process exactly 512 token IDs")

    def test_thread_mismatch_fails(self) -> None:
        records = valid_records()
        target = next(
            record
            for record in records
            if record["platform"] == "x86_cpu"
            and record["workload"] == contract.QWEN_PP512
            and record["engine"] == "llama.cpp"
        )
        target["threads"] = 6
        self.assert_fails(records, "native and llama.cpp differ in threads")

    def test_multiple_visible_gpus_fail(self) -> None:
        records = valid_records()
        target = next(record for record in records if record["platform"] == "cuda")
        target["device"] = {
            "kind": "cuda",
            "id": "0",
            "gpu_count": 2,
            "visible_gpu_ids": ["0", "1"],
        }
        self.assert_fails(records, "GPU benchmarks must enumerate exactly one GPU")

    def test_median_below_gate_fails(self) -> None:
        records = valid_records()
        for record in records:
            if (
                record["platform"] == "metal"
                and record["workload"] == contract.EMBEDDING_ENCODER
                and record["engine"] == "native"
            ):
                _rewrite_rate(record, 104.0)
        self.assert_fails(records, "median speedup 1.040000x is below 1.05x")

    def test_one_native_sample_below_llama_median_fails(self) -> None:
        records = valid_records()
        target = next(
            record
            for record in records
            if record["platform"] == "apple_cpu"
            and record["workload"] == contract.QWEN_PP512
            and record["engine"] == "native"
            and record["sample_index"] == 0
        )
        _rewrite_rate(target, 99.0)
        self.assert_fails(records, "slowest native sample is 0.990000x")

    def test_target_only_native_tg_is_rejected(self) -> None:
        records = valid_records()
        target = next(
            record
            for record in records
            if record["workload"] == contract.QWEN_TG128 and record["engine"] == "native"
        )
        target["generation_path"] = "target_only"
        self.assert_fails(records, "native TG128 generation_path must be production_mtp")

    def test_different_tg_output_ids_fail(self) -> None:
        records = valid_records()
        target = next(
            record
            for record in records
            if record["platform"] == "cuda"
            and record["workload"] == contract.QWEN_TG128
            and record["engine"] == "llama.cpp"
            and record["sample_index"] == 2
        )
        output_ids = list(target["output_token_ids"])
        output_ids[-1] += 1
        target["output_token_ids"] = output_ids
        target["output_token_ids_sha256"] = contract.token_ids_sha256(output_ids)
        self.assert_fails(records, "committed TG128 token IDs differ")

    def test_embedding_generation_tokens_are_rejected(self) -> None:
        records = valid_records()
        target = next(record for record in records if record["workload"] == contract.EMBEDDING_ENCODER)
        target["output_tokens"] = 1
        self.assert_fails(records, "embedding encoder output_tokens must be 0; TG is not applicable")

    def test_missing_platform_fails_by_default(self) -> None:
        records = [record for record in valid_records() if record["platform"] != "cuda"]
        self.assert_fails(records, "missing required platforms: ['cuda']")

    def test_hardware_hash_mismatch_fails(self) -> None:
        records = valid_records()
        target = next(
            record
            for record in records
            if record["platform"] == "metal" and record["engine"] == "llama.cpp"
        )
        target["hardware"] = {**target["hardware"], "memory_bytes": 64 * 1024**3}
        target["hardware_sha256"] = contract.hardware_sha256(target["hardware"])
        self.assert_fails(records, "native and llama.cpp must run on identical hardware")

    def assert_fails(self, records: list[dict[str, object]], expected: str) -> None:
        with self.assertRaises(verify.VerificationError) as raised:
            verify.verify_records(copy.deepcopy(records))
        self.assertIn(expected, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
