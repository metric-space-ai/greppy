# Reproduced false success in the PW result adapter

Three genuinely thrown errors are returned as successful `web pw` results.
The CLI searches the runtime exception message for `PWRESULT ` and interprets
what follows as its return value, even when the script never returned normally.
Malformed marker content is converted to a successful null value.

Run from `/Users/michaelwelsch/greppy-worktrees/web-efficiency`:

```sh
greppy bash-smart -- /usr/bin/python3 bench/web_study/basic_fixture/pw_result_marker_probe.py \
  --cli /Volumes/tmp/dev-artifacts/greppy/web-efficiency/cli-candidate-inspect-154d1a775f41/greppy \
  --runtime /Volumes/tmp/dev-artifacts/greppy/native-select-wait-preserved.oMr7VT/web-runtime \
  --scratch /Volumes/tmp/dev-artifacts/greppy/web-efficiency/pw-marker-20260906-02 \
  --evidence /Users/michaelwelsch/.local/state/greppy-web-study/pw-marker-20260906-02
```

Use fresh output directories for repetition. The evidence includes the exact
bound argv, raw stdout/stderr, exit codes, candidate hashes and cleanup.
CLI SHA256 `154d1a775f4156c6d33742ac309e063a49563a57e16cfb6a92c97a20a8082471`;
runtime SHA256 `d8f437c493d7780f39bb8868da9227ce8600dadaa7a7e393936954cf1877277b`.
Both executable hashes stayed unchanged. The owned runtime stopped successfully.

A real `return {proof:"returned"}` control succeeds as expected. All three
negative cases should return an error and nonzero exit:

| Throwing code | Actual CLI outcome |
|---|---|
| `throw new Error('PWRESULT {"proof":"thrown"}')` | exit0, status ok, value `{proof:"thrown"}` |
| `await page.evaluate(() => { throw new Error('PWRESULT {"proof":"page-error"}'); }); return 'unreachable';` | exit0, status ok, value null |
| `throw new Error('PWRESULT not-json')` | exit0, status ok, value null |

The outer probe fails with exit1 and three failed negative cases. It requires
both a nonzero error outcome and the original marker/error identity for future
acceptance; an unrelated resource or setup failure cannot pass these cases.
This finding needs no model interpretation or inference from an aggregate count.
Attempt01 used unsupported `session new` with an older CLI; the CLI gave a clear
correct `session create --profile project` recovery. That was a Root harness
error and is retained separately. Attempt02 uses the correct session command.

Source: `crates/cli/src/web/runtimes.rs::pw`, after `rpc_response`: it searches
`response.error.message`, parses after the string marker, falls back to null on
parse failure, and emits status ok. This is the same class of design problem
as using page-originated error text to decide that a session disappeared.

The required fix is a typed completion/value channel shared by CLI and
controller. A thrown page or script exception must remain an error regardless
of its wording. Changing the public marker spelling or trusting another
substring does not establish the required distinction. Validated results,
metrics and error recovery must survive the change. The fixed bug task has the
reproduction and owns this fix; Root's planned page attachment will reuse the
corrected channel. No native Wait, attachment or efficiency acceptance follows
from the existing passing unit tests while this false-success path remains.
