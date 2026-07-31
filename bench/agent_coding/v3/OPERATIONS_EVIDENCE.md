# Signed operational evidence

Productive V3 runs use an operations-owned RSA or EC public key. Detached
signatures are verified with `openssl dgst -sha256 -verify`; private keys never
enter the runner host command line or result archive.

## Preflight attestation

Run `preflight_gpu3.py`, retain its exact output bytes, then create and sign:

```json
{
  "schema_version": "greppy.agent-coding-v3.gpu3-preflight-attestation.1",
  "ready": true,
  "preflight_report_sha256": "<sha256 of exact report bytes>",
  "runtime_bindings": {"<exact object emitted by the runner preflight helper>": "<values>"}
}
```

`runtime_bindings` contains hashes for runner source, gate contract, pricing
contract, Greppy binary, `AGENTS.md`, provider extension, network attestation
and audit proof, plus provider/model, live image ID and the aggregate hash of
all read-only dependency trees. The runner compares the object for exact JSON
equality; a changed binary, manual, harness, model, image, proxy audit, price,
gate or dependency cache invalidates it.

Sign the exact attestation bytes:

```bash
openssl dgst -sha256 -sign /offline/operations-private.pem \
  -out preflight-attestation.sig preflight-attestation.json
```

## Smoke evidence

After running and manually reviewing exactly three paired tasks (six arm
traces), create and sign:

```json
{
  "schema_version": "greppy.agent-coding-v3.smoke-evidence.1",
  "ready": true,
  "paired_trajectory_count": 3,
  "arm_trace_count": 6,
  "task_ids": ["task_...", "task_...", "task_..."],
  "smoke_run_archive_sha256": "<sha256>",
  "arm_trace_sha256": ["<six exact trace sha256 values>"],
  "manual_review": {
    "passed": true,
    "read_all_six_arm_traces": true,
    "open_findings": []
  },
  "preflight_attestation_sha256": "<sha256 of signed preflight attestation>",
  "runtime_bindings": {"<same exact object>": "<values>"}
}
```

Sign it with the same detached-signature command. The full runner refuses
missing, unsigned, non-three-pair, open-finding or stale smoke evidence. A task
subset remains smoke-only and records `release_gate.passed` as `null`; it is not
an accidental failed or successful release. Once the credential broker enables
full execution, a full run records the decision, writes the archive even on
statistical failure, and exits `3` when the release gate is false.

The runner archive and resume checkpoint both bind the operational evidence,
runner/gate/pricing hashes and read-only dependency identities. Resume therefore
fails closed after any of these inputs changes.

Operational signatures do not waive the credential boundary. With the current
in-process provider key, only subset smokes can execute and they are not valid
cost evidence. A full run remains blocked until a broker prevents agent child
processes from reading the key or issuing unattributed provider calls.
