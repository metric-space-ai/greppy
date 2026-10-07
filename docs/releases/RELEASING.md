# Releasing greppy

A release is one commit that passed every gate, tagged `vX.Y.Z`. Pushing the
tag runs `.github/workflows/release.yml`, which builds, signs, verifies and
publishes the packages. Nothing is uploaded by hand.

1. **Version and changelog.** Bump `version` in the workspace `Cargo.toml` and
   date the `CHANGELOG.md` section on the release commit.
2. **Gates on the exact commit.** `ci.yml`, `codeql.yml`, `security-audit.yml`
   and `filesystem-cow.yml` must each have a successful run on the commit;
   the release workflow checks this again before it builds anything.
3. **Signing dry-run.** Dispatch `release.yml` on the branch with
   `sign_dry_run=true`. It builds, signs and notarizes the macOS package and
   runs the clean-package checks (Metal inference smoke and daemon stress on
   macOS, structural checks on Linux) without publishing.
4. **CUDA acceptance.** Hosted runners have no GPU and product builds refuse
   CPU inference, so the Linux package's inference is checked on a GPU host:

   ```sh
   tools/release_cuda_acceptance.sh <dry-run-id> <ssh-host-with-nvidia-gpu>
   ```

5. **Tag.** `git tag -a vX.Y.Z <commit> -m "greppy X.Y.Z"` and push the tag. A
   tag is used once: a force-updated or deleted tag is rejected, and a release
   that already exists is never replaced.

Notes

- The Linux runtime footprint is not measured on hosted runners (no GPU); the
  macOS Metal footprint is measured on every tag.
- Summary-quality evidence is attached when `summary-quality.yml` has a
  passing run on the commit; without one the release is published without it
  and the workflow says so.
- Agent benchmarks find defects for the next release; they never gate one.
