# ACP host integration

`greppy agent stdio` exposes the coding agent to an ACP host such as Workjet.
It uses newline-delimited JSON-RPC on stdin/stdout. Human diagnostics go to
stderr. The host launches a dedicated process and supplies the project or
worktree folder in `session/new` or `session/load`.

```sh
greppy agent stdio --endpoint http://127.0.0.1:8317 --model MODEL_ID
```

`--max-turns` bounds one prompt (default40 assistant turns). The endpoint is an
Anthropic Messages-compatible gateway. Authentication reuses `GREPPY_API_KEY`
from the agent process; ACP clients do not transmit or persist a secret. Model
selection can be supplied with `--model`, `session/set_model`, or the `model`
option in `session/set_config_option`.

The host controls the working folder. ACP tools execute in that folder after
client permission; the host should supply its intended isolated worktree when
isolation is needed. The ACP entry point does not create or automatically apply
a one-shot proposal from `refs/greppy/agent/`.

## Supported operations

| Operation | Behavior |
| --- | --- |
| `initialize` | Negotiates protocol1 and reports capabilities/version. |
| `authenticate` | Accepts the advertised `greppy.env` method. |
| `session/new` | Creates a persisted session for an existing folder. |
| `session/load`, `session/resume` | Loads saved history and recorded model; emits replay updates. |
| `session/list` | Lists persisted sessions for the supplied folder/project. |
| `session/prompt` | Accepts text content blocks and streams agent/tool updates. |
| `session/set_model`, `session/set_config_option` | Stores the selected model for subsequent turns. |
| `session/cancel` | Cancels model reads and permission waits; an executing tool finishes at its safe boundary. |
| `session/close` | Closes an idle session. |

The agent sends `session/update` text, thought, and tool events. Before a tool
executes it sends `session/request_permission`; only a selected advertised
allow option permits execution. Rejection, cancellation, missing responses on
connection loss, and unrecognized options deny execution. Permission memory
and cancellation are scoped to the session. Closing stdin cancels outstanding
prompts and permission waits before process shutdown.

Session history and model metadata use Greppy's existing session store. Each
session captures its data root and logical project when created or loaded;
concurrent prompts do not change process-wide project environment variables.
Completed turns stage messages, cumulative usage, stop reason, and initial title
in a private file, then replace the session log atomically. Failed staging keeps
both saved and live history unchanged. A stable per-session writer lease spans
validation and replacement across processes; competing writers fail explicitly,
and append/model writes use the same lease. A restarted host can load the same
session id. Replayed updates carry `_meta.isReplay` so the host can distinguish restored
history from a new turn.

## Wire shape and limits

Requests and responses use JSON-RPC2.0. Notifications accept the standard
missing id and Workjet/Effect's empty-string id. Outbound notifications include
an empty id and headers array for that client. Typed failures include the
Effect `Cause`/`Fail` representation as well as the JSON-RPC error code/message.

Images, audio, embedded resource context, externally supplied MCP servers,
forking, and pagination cursors are unsupported and fail explicitly. A session
cannot receive overlapping prompts. Stream size/event caps remain those of the
existing Messages client.

This draft integration still requires compilation, loopback regression
execution, and a real Workjet ACP-client test before release. The Workjet
provider adapter must launch this entry point and configure the gateway/model;
adding only a menu label does not establish a working harness.
