# Qwen3.5 function-purpose finetuning package

This directory is the secret-free training package captured on 2026-07-13 for
the Greppy Qwen3.5-0.8B function-purpose model. It publishes the exact pipeline
programs and a reconstruction environment, but not the private training rows or
generated labels. `SOURCE_SCRIPTS.sha256` proves the eight files in `scripts/`
are byte-identical to the captured operational snapshots.

## Published and withheld material

Published here:

- exact teacher-pilot, production generation, SFT rebuild, full-parameter SFT,
  MTP, continuation/export, MTP-strip, and checkpoint-promotion scripts;
- an exact-version Python dependency lock and a sanitized machine/runtime lock;
- frozen input, model, tokenizer, config, export, and script hashes;
- a standard-library private-data auditor and standard-library unit tests.

Not published:

- raw function source rows, repository paths/commits for individual selected
  rows, holdout source, or generated MiniMax labels;
- private JSONL files (`week*_raw.jsonl`, rebuilt SFT JSONL, and fixed eval
  JSONL), API credentials, model checkpoints, or generation logs.

The audit operates locally on those private files and emits aggregate counts,
repository/language/license histograms, and one aggregate digest. It never emits
source text, labels, or per-row source/label hashes.

## Immutable inputs and revisions

Base model: `Qwen/Qwen3.5-0.8B` at revision
`2fc06364715b967f1860aea9cf38778875588b17`.

- model SHA-256: `04b1c301e7789b32eddea47531cb8ddd3c690e6bc29e0911b14f7c7280544696`
- tokenizer SHA-256: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- config SHA-256: `b90b86eaf5e4810a99348c3d271b7516f93d1529ab23add6dcf189076a512204`

Source dataset: `Fsoft-AIC/the-vault-function` at revision
`505c679056e49a2a269b64777ee7c496d22e1440`. The [dataset card at that exact
revision](https://huggingface.co/datasets/Fsoft-AIC/the-vault-function/blob/505c679056e49a2a269b64777ee7c496d22e1440/README.md)
states `license: mit`, states that The Vault derives from The Stack's
permissively licensed source code, and documents a per-row `license` metadata
field. These are statements from the pinned card, not a legal conclusion about
any particular row or generated output.

Frozen ordered training inputs:

| Input | Rows | SHA-256 |
| --- | ---: | --- |
| Initial stage | 4,509 | `4b378e6fe10cec79ae76176e5e2b5fce9637960928080f286fe20840cf24852d` |
| Continuation 1 | 30,651 | `ceac48839c769123a7ca8e40baffc282ac26816739e879a471921cc08d69192a` |
| Continuation 2 | 20,114 | `a2d75ee1e39065666657cd3baa86d09b7119921567916d9479387b94f7ff5aeb` |
| Rebuilt aggregate | 55,274 | `3e0badd041cf93e5d8e2d0e0b7e983f941562dd68ca94c7592884eae69946ff3` |
| Fixed eval | 987 | `02d42760f44c3d410c43f41675ddcd2c4ee2916c0f2cad2d25049466ea0f186e` |

The stage row counts total 55,274. Hashes identify the ordered JSONL bytes, so
line order, JSON serialization, prompts, and labels are part of each identity.

The 2026-07-13 continuation freezes the historical 55,274 rows together with
214,105 newly selected rows:

| Input | Rows | SHA-256 |
| --- | ---: | --- |
| Full ordered SFT snapshot | 269,379 | `a79fef839541110a05ad0712565e2f05a883d96e4ac644e20d86e674e2bfca18` |
| New continuation tranche | 214,105 | `778c8c436cc81a0df78be1a9c44e9afea345c6a2c4080c95574e12d09d5f97f1` |

Token-length filtering retained 211,726 continuation rows and 984 fixed-eval
rows at `MAXLEN=1280`. The continuation starts from the published 2026-07-11
checkpoint identified in `provenance.json`; its output hashes remain pending
until training, evaluation, export, and quantization finish.

Teacher labels were generated with MiniMax-M3 and prompt version
`minimax-m3-brief-K-v1`, whose exact template is `PROMPTS["K"]` in
`scripts/pilot_run.py`. MiniMax's [Open Platform Terms of
Service](https://platform.minimax.io/protocol/terms-of-service) state an
effective date of **March 30, 2026**. This package records the cited page and
date as factual provenance only; it makes no conclusion about rights,
compliance, or redistribution. Raw data and labels remain unpublished.

The student prompt is version `qwen35-brief-v3`, exactly:

```text
<|im_start|>user
brief:
{source}<|im_end|>
<|im_start|>assistant
```

## 2026-07-13 training launch

The current 2026-07-13 continuation uses three distributed processes/GPUs,
per-device batch size 1, gradient accumulation 8, maximum sequence length
1,280, and one epoch. The resulting global batch before sequence-length
variation is `3 * 1 * 8 = 24`. Linux CPU affinity is restricted to P-core
logical CPUs `0-11`. Parameters are loaded in FP32 and retained as FP32
masters; `bf16=True` enables BF16 autocast. The MTP block is jointly trained in
FP32 with loss weight `0.2` and shared trunk embeddings/output head. Historical
stages used the launch settings recorded by their source scripts and logs; this
section does not retroactively assign the current three-GPU shape to them.

The launch shape was:

```bash
taskset -c 0-11 env \
  CUDA_VISIBLE_DEVICES=0,1,2 \
  SMOKE_BS=1 \
  SMOKE_ACCUM=8 \
  SMOKE_EPOCHS=1 \
  SMOKE_INIT="$INIT" \
  SMOKE_DATA="$DATA" \
  SMOKE_OUT="$OUT" \
  "$VENV/bin/python3" -m torch.distributed.run \
    --nproc_per_node=3 scripts/train_smoke.py
```

`INIT`, `DATA`, and `OUT` selected the base/previous stage, the corresponding
frozen JSONL, and the new checkpoint. Other exact training settings are in the
unchanged script: Adafactor, learning rate `1e-5`, cosine schedule, warmup ratio
`0.03`, gradient checkpointing, epoch evaluation, random seed `3`, and no
intermediate Trainer save. `MAXLEN=1280` is hard-coded. The preserved
`daily_cycle.sh` also contains a later two-GPU daily automation default; that is
not the three-GPU launch recorded above.

## Environment and reconstruction

The source environment capture mixed the training environment with unrelated
desktop, notebook, server, and Ubuntu-managed packages and contained
host-specific GPU identifiers and live telemetry. `requirements.lock` retains
exact versions for the training, data, and GGUF dependency closure;
`environment.lock.json` retains the relevant OS/CPU/CUDA/runtime facts while
removing host name, PCI addresses, UUIDs, serials, temperatures, clocks, and
utilization. The source capture did not include the Python interpreter version,
which is recorded explicitly rather than guessed.

Create an isolated environment and install the lock:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --requirement requirements.lock
```

The exact generator uses `main` in its Hugging Face download URLs and contains
historical absolute host paths. For a pinned reconstruction, materialize The
Vault shards from revision `505c679...` into the expected local vault directory
before running `prod_gen.py`; its existing-file path avoids a new `main`
download. Set up the historical directory layout or invoke the copied modules
from an equivalent layout. The frozen JSONL hashes, rather than a fresh teacher
API run, define the training inputs. Because raw rows and labels are withheld,
this public package alone cannot regenerate those private JSONL bytes.

## Export

The merged BF16 safetensors checkpoint SHA-256 is
`237cf0a76d57d4c4e6fdbbcc5af0020f9413060b8b1ccb145a95b4c75368f11e`.
GGUF conversion and `Q4_K_M` quantization used `llama.cpp` revision
`ef2d770117db45b05aa7ecd1b0acca36370c5470`. The exact conversion,
quantization, and optional MTP-strip command sequence is preserved in
`scripts/daily_cycle.sh`; `scripts/strip_mtp.py` is the exact GGUF rewrite.
The released MTP-bearing GGUF SHA-256 is
`d45e08ad7bb8787ae9b6f56b6915e8b44ac6e13c6b740fdc7bd591249209a72c`.

## Private-data audit

Run the audit where the unpublished files exist:

```bash
python3 audit_sft.py \
  --raw /private/prod/week*_raw.jsonl \
  --sft /private/frozen/all_sft.jsonl \
  --output /private/audit/qwen35-sft-audit.json
```

The shipped `summary_quality_holdout_repos.txt` is always active and contains
these canonical upstream identities:

- `serde-rs/serde`
- `pallets/flask`
- `google/gson`
- `colinhacks/zod`
- `tokio-rs/tokio`
- `gohugoio/hugo`

Repository matching is case-insensitive and normalizes common GitHub URL and
`.git` spellings. Any selected SFT row from a denied repository fails the
audit. Additional newline-delimited `owner/repo` lists can be supplied with
repeated `--denylist PATH` arguments.

The 2026-07-13 private audit mapped all **269,379 of 269,379 selected SFT
rows** to accepted raw audit records and found **zero selected rows** matching
those six repositories. It covered 269,625 raw rows, 269,380 accepted rows,
245 dropped rows, 54,324 repositories, 325 license combinations, and one
byte-identical source duplicate that the selected SFT row disambiguates by its
repo/language provenance. The deterministic aggregate row digest is
`fa10bae3a37e19cda295394155199ae85a5d8240131060f49e5284a14eae384c`.
See `audit-summary-2026-07-13.json` for the compact machine-readable result.
The complete 54,324-entry repository histogram and 325-entry license
histogram are published as deterministic gzip in
`audit-report-2026-07-13.json.gz`. Its compressed SHA-256 is
`aed01649bbf9d671c7a225d6ce684c85f3ac8a6b20a678920591094cbd3b1286`;
the uncompressed JSON is the report already anchored by SHA-256
`ad93aaa5e2982c4d04da0d744c16bd7226845c682b6a3f03742da699d1de89cc`.

This is a recorded result of that frozen private audit, not a claim about
future data. The auditor independently fails on an unmapped SFT prompt, any
distinct accepted prompt absent from SFT, ambiguous raw/SFT duplicates,
SHA-256 or prompt-normalization collisions, missing/invalid licenses,
provenance or label mismatches, unexpected schemas, and selected denylist
matches. The full aggregate report is now public; raw rows and labels remain
withheld.

## Verification

From this directory:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q .
python3 verify_package.py
```

`verify_package.py` checks exact source-script hashes, complete package-manifest
coverage, JSON syntax, and every package file against high-confidence private
key and service-token signatures. `MANIFEST.sha256` covers every package file
except itself.
