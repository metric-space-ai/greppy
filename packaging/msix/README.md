# Microsoft Store MSIX packaging for greppy

Store product: **greppy**, Store Product ID **9NNRV0BJ550L**
(<https://apps.microsoft.com/detail/9NNRV0BJ550L>).

Distribution model: we produce an **UNSIGNED** `.msix`; the **Microsoft Store
signs the submission itself** after certification. No Authenticode secrets are
involved anywhere in this pipeline (owner decision
`WINDOWS_SIGNING_MODE=MICROSOFT_STORE_MSIX`, mirrored in `release.yml`).

## Files

| File | Purpose |
| --- | --- |
| `AppxManifest.xml.in` | Manifest template for a packaged win32 **console** app: `Windows.Desktop` target, `runFullTrust`, and an `AppExecutionAlias` so `greppy.exe` is on `PATH` after Store install. Placeholders `@IDENTITY_NAME@ @PUBLISHER@ @PUBLISHER_DISPLAY@ @VERSION@ @ARCH@` are rendered by `tools/build_msix.ps1`. |
| `identity.json` | Package Identity config. The three `TODO` fields must be copied **verbatim** from Partner Center (see below). |
| `assets/*.png` | Placeholder monochrome "g" tiles (Square44x44, Square150x150, StoreLogo). **Final art is a later owner task** — replace these PNGs in place; sizes must stay 44x44 / 150x150 / 50x50 (or add scale-qualified variants). |
| `../../tools/build_msix.ps1` | Renders the manifest, lays out `greppy.exe` + assets + license notices, runs `makeappx pack`, validates the packed manifest, emits `greppy-<version>-<arch>.msix`. |
| `../../.github/workflows/msix.yml` | `workflow_dispatch` CI build on `windows-latest` (x64) and `windows-11-arm` (arm64). |

## One-time: fill in the package identity

1. Partner Center → **Apps and games** → greppy → **Product management** →
   **Product identity**.
2. Copy these three values into `packaging/msix/identity.json`, replacing the
   `TODO:` strings exactly (no trimming, no case changes):
   - `Package/Identity/Name` → `identity_name`
   - `Package/Identity/Publisher` (a `CN=GUID` string) → `publisher`
   - `Package/Identity/PublisherDisplayName` → `publisher_display_name`
3. Commit. The build fails loudly while any `TODO` remains — except when the
   workflow input `allow_placeholder_identity: true` is used to smoke-test the
   pipeline (that output is **not submittable**).

## Build

CI (preferred): GitHub → Actions → **msix** → *Run workflow*. Download the
`greppy-msix-x64-<sha>` / `greppy-msix-arm64-<sha>` artifacts; each contains
`greppy-<version>-<arch>.msix`.

Locally on a Windows machine with the Windows 10/11 SDK:

```powershell
cargo build --locked --release --bin greppy   # after ./tools/fetch_model_assets.sh
./tools/build_msix.ps1 -Arch x64 -OutputDir msix-out
```

The MSIX version is the Cargo workspace version with a `.0` revision appended
(the Store reserves the 4th part and requires it to be `0`;
<https://learn.microsoft.com/en-us/windows/msix/package/app-package-requirements>).

## Submit in Partner Center

1. Partner Center → greppy → **Start/Update submission** → **Packages**.
2. Upload the unsigned `.msix` for each architecture (x64, arm64). Upload
   both into the same submission so the Store serves the right one per device.
3. Availability/pricing/listing as desired; the listing **must clearly say
   this is a command-line tool** launched from a terminal (see certification
   notes) and describe usage (`greppy --help`).
4. In the submission's **Restricted capabilities** section, justify
   `runFullTrust`: "greppy is a packaged win32 command-line developer tool; it
   reads and indexes source repositories on the local file system and runs
   fully local ML inference. All packaged desktop apps require runFullTrust."
5. Submit for certification. Microsoft signs the package after it passes.

## Certification notes for CLI apps

- **AppExecutionAlias** (`greppy.exe`) is what makes the tool usable: after
  install it appears in `%LOCALAPPDATA%\Microsoft\WindowsApps` which is on
  the user `PATH`. Users open a new terminal and run `greppy`.
- The manifest hides the Start-menu tile (`AppListEntry="none"`), which is
  allowed; testers must still be able to discover functionality, so the Store
  description has to state that the app is used from a terminal and give a
  first command to run. Consider adding a note for testers in the submission's
  "Notes for certification" box: "Command-line tool. After install, open
  Windows Terminal and run `greppy --help`."
- `runFullTrust` triggers a manual capability review; the justification in
  step 4 above has been sufficient for comparable CLI tools.
- The binary embeds ML models; the package also ships the `licenses/` notices
  (Gemma/Qwen), same as the direct-download archives.

## Local install testing (self-sign — never for the Store)

The Store-bound package stays unsigned. To *install locally* for testing you
must sign it with a self-signed cert whose Subject equals the manifest
Publisher, then trust that cert:

```powershell
$publisher = (Get-Content packaging/msix/identity.json | ConvertFrom-Json).publisher
New-SelfSignedCertificate -Type Custom -Subject $publisher -KeyUsage DigitalSignature `
  -FriendlyName 'greppy msix test' -CertStoreLocation Cert:\CurrentUser\My `
  -TextExtension @('2.5.29.37={text}1.3.6.1.5.5.7.3.3', '2.5.29.19={text}')
# Export-PfxCertificate ... ; then:
signtool sign /fd SHA256 /f greppy-test.pfx /p <password> greppy-<version>-x64.msix
# Import the cert into LocalMachine\Trusted People, then:
Add-AppxPackage greppy-<version>-x64.msix
greppy --version   # in a NEW terminal (alias registered at install)
```

Uninstall with `Remove-AppxPackage` (find it via `Get-AppxPackage *greppy*`).
**Never upload a self-signed package to Partner Center.**

## Open owner tasks

- [ ] Fill `identity.json` from Partner Center (build is gated on it).
- [ ] Replace placeholder tile art in `assets/` with real branding.
- [ ] First submission: complete listing, age ratings, and the
      `runFullTrust` justification.
