# Provider credential boundary

The current Pi/provider process must receive the MiniMax key to make model
requests. The runner mounts a mode-0600 ephemeral key file read-only, reads it
inside the container, redacts its exact bytes from captured stdout/stderr and
rejects it in the agent diff. The proxy, audit container, Docker labels and
published reports never receive the key.

This does **not** isolate the key from agent-launched child processes. Pi and
shell tools share one container/process trust domain, so a malicious or
prompt-injected shell command could inspect the provider environment. Prompt
instructions are not a security control, and the V3 evidence states
`agent_child_process_can_read_provider_environment: true`.

A real fix requires moving authenticated provider calls into a separate broker
that the agent can invoke but cannot introspect, with request-shape validation,
rate/usage attribution and no generic credential-bearing environment in the
agent container. That design is future work. V3 subset smokes may use a scoped,
short-lived, spend-limited benchmark key, but their archive is marked
`cost_gate_valid: false`. The runner hard-blocks a full 144-task invocation
before its first agent trajectory. There is no Boolean risk-acceptance bypass:
enabling a full run requires a broker implementation plus verifiable broker
attestation, which also changes the runner binding and invalidates old smoke
evidence.
