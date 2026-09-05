# Optional native agent model preparation

The local agent is an optional alternative to the existing localhost Messages
gateway. This change prepares reusable Rust components; it does not advertise
a working local CLI backend, ship weights, download a model, or admit an
unfinished executor.

## First-use contract

- Installation, help, ordinary Greppy navigation/indexing, model listings and
  gateway agent sessions must not download or prewarm the agent model.
- Only the first actual request through the explicitly selected local agent
  backend may provision missing assets and start/load its engine.
- Reuse a persistent per-user model cache across repositories, processes and
  daemon restarts. Model TTL releases RAM/VRAM, not cached weights.
- Publish a downloaded object only after exact length and SHA-256 verification.
  Serialize downloads of the same digest. Incomplete transfers are never ready.
- Distinguish missing assets, waiting for another downloader, download progress,
  artifact verification, signed release admission, backend loading and ready.
  Receiving the final download byte does not mean the model is usable.
- A user cancellation must stop download or generation. Do not silently replace
  the requested backend with the gateway or another hardware backend.

## Implemented preparation

`greppy_agent::local_model::ArtifactCache` is a content-addressed blob cache.
Its constructor and verified lookup never fetch. `ensure_with` is the explicit
first-use entry point: an injected fetcher writes with bounded memory into a
same-filesystem staging file. Size/SHA-256 checks precede atomic publication.
An OS lock serializes same-object downloads across clients. Warm cache lookup
rehashes bytes; do not use it for frequent UI status polling.

The root is supplied as an absolute persistent path. Test fixtures and build
caches belong on disposable development storage; installed user model assets
must not default there. Normal transfer failures clean their staging files;
hard-crash staging leftovers are ignored. Resume/range requests and crash-file
garbage collection are not implemented in this preparation. Lock acquisition
is blocking: use a worker thread, and add cancellable lock waits and waiting
progress before connecting this primitive to an interactive first-use flow.

`OnDemandModel` implements the existing `ModelStream` contract. Constructing,
inspecting or dropping an unused adapter does not invoke its factory. The first
turn invokes the factory, which will provision the signed release and create a
ready engine adapter. Successful turns reuse it. Failed initialization remains
retryable; a failed turn drops the adapter rather than reusing partial session
state. The real adapter must reset/cancel/release its session on failure/drop.

Tests use tiny byte fixtures and a scripted model through the real agent loop
to cover first use, tool-result history, cancellation before first use, cache
reuse, corruption, interrupted and oversized transfers, concurrent clients and
failed initialization. They are not model-quality or hardware tests.

## Engine integration boundary

The existing `ModelStream`, `ModelRequest`, `StreamEvent` and `TurnResult`
are already provider-neutral. Keep agent history and host-side tool execution
there; do not add another agent loop or another tokenizer/sampler.

The inspected CTOX-LLM contract is model-local:
`models/qwen38_27b/docs/ENGINE_ABI_V1.md` and `WIRE_PROTOCOL_V1.md`.
It exposes `EngineServer::load_signed` over the same native engine lifecycle,
including warmup, prefill/decode, cancellation, reset, health and unload.

The next adapter must:

1. Resolve an authenticated release catalog to a release ID, backend pack and
   memory profile. Download the complete selected release asset closure,
   including pinned tokenizer/template and restricted MTP vocabulary. The blob
   cache does not authenticate a publisher or assemble a release tree.
2. Pass one verified installation and trusted signing key to
   `EngineServer::load_signed`. Preserve executor promotion, context capacity,
   explicit backend choice and no-hidden-fallback admission checks. Do not
   hardcode an unreleased training candidate SHA or invent a release URL.
3. Own the engine in Greppy's daemon process, keeping CUDA state on its dedicated
   executor thread. Reuse lifecycle/owner/status concepts from
   `inference_daemon.rs`. Its current single-JSON-response handler cannot carry
   the engine's multi-event streaming protocol unchanged.
4. Use versioned JSON Lines over Unix sockets or Windows named pipes. Keep
   request, operation and session identities distinct; send cancellation on a
   separate connection so it can interrupt active decode. Do not introduce a
   managed HTTP localhost gateway.
5. Map full system/user/assistant/tool history and tool schemas through the
   engine's pinned Responses frontend. Preserve ordered text/thinking/tool
   events and usage/stop reasons. Do not execute a partial tool-call payload.
   Reject unsupported images explicitly. Generation is currently greedy;
   non-greedy MTP is not admitted.
6. Stream accepted MTP-prefix tokens before the target bonus/fallback exactly
   once. Reset invalid sessions and surface cancellation distinctly. Respect
   the engine's unload residue checks before claiming memory was released.

The localhost client remains unchanged. CLI/TUI backend selection, HTTPS
fetching/retries/cancellation, release signature/catalog verification, release
assembly, daemon streaming transport and the actual CTOX executor dependency
remain follow-up work. These primitives alone do not enable local inference.

## Delivery sequence and hardware evidence

First demonstrate one complete local CUDA coding/tool task, streaming and
cancellation, using an explicitly authorized verification/development assembly
until the engine's production gates pass. Do not bypass signed admission to
make an unpromoted executor appear production-ready.

Then establish the same functional contract on CPU, Metal and Android/Snapdragon,
followed by platform-specific kernel and memory optimization. CPU correctness
execution exists; production SIMD/full graph, Metal and Snapdragon execution
remain engine work. Mobile GPU/NPU selection must follow capability measurement.

The target pack is Qwen3.8-27B with mixed Q2/Q4, calibrated s_in/s_out scales and
resident MTP. The reported 8,373,658,112-byte candidate is not a final release or
proof of less than 10 GB peak VRAM. Measure full resident model, KV/recurrent
state, MTP, scratch and runtime overhead for every accepted context profile.
The Android target has a stated 16 GB shared-memory budget, including OS/app
overhead, and needs sustained thermal/energy measurements.

Track prefill and decode separately alongside correctness, coding/tool success,
peak allocated/driver memory, cold/warm load and unload residue. Optimization
comes after a real functional baseline on each backend.
