# Native source integration checkpoint

After S08's ten participants and observers completed, the worker's prerequisite
slice was applied to the efficiency worktree. The slice replays commits
9690177d, c9e97106, ebd203b6, 30c4e3db, dd108d54, 5c39616c, 9c8e8fa2,
8da50c22, fdd4208d and 6a22a0aa. Already equivalent Root additions and the Root
attachment component are preserved; absent historical release notes stay absent.

Patch: `/Volumes/tmp/dev-artifacts/greppy/native-integration-check.q2h1ez/integration.patch`,
SHA256 `ac8d8cc774bd8f44a00311606c546c871bfe531429a05ee52ad96bdc8a1cb852`.
All 27 applied files match the reviewed replay tree
`a759037f23a59cab9e599456e6191619f0773ae1` byte for byte. Receipt:
`/Volumes/tmp/dev-artifacts/greppy/web-efficiency/native-integration-source-20260906.json`.

The reviewed changes cover bounded capability startup, stable document/node refs,
form-state observations, truthful select errors, strict bounded waits across
navigation, action state attached to the resolved session/tab, and preservation of
other fresh image digest caches. They do not constitute an incremental native DOM
index, an attached Playwright API, or a completed Page.url fix.

Validation in this actual Root tree:

- 88 CLI web unit tests passed, no guard termination and unchanged listed source
  hashes: `cli-guarded-h3ls_498/receipt.json` in the efficiency artifacts directory.
- 25 JavaScript component tests passed for observed refs, option selection and
  wait predicates, including stale-ref absence refusal and page-handler reversal.
- `git diff --check` passed.

The preserved native runtime had already passed worker native regressions and
Root's actual CLI-driven table and wait/ref preflights. It was not rebuilt from
this newly integrated Root tree. A fresh native source build/full native suite and
signed release acceptance remain separate and open. Disk space is insufficient
for an unrestricted new native build; no foreign or frozen artifact was removed.

The installed Greppy patch parser refused the valid Git envelope with exit20 and
an unlocated prefix error; no files were changed by that attempt. The defect was
reported exclusively to the fix worker, and the verified `git apply` alternative
was used. Worker reproduction confirms the installed parser issue is already
fixed in7b45519a/45bd10c1; new-file support remains a separate explicit API limit.
No duplicate parser fix was introduced here.

The new S08 error-view and recovery-guidance findings remain separate follow-up
fixes. They were not silently folded into the frozen experimental candidates.
