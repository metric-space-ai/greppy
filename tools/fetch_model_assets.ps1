# Windows counterpart of tools/fetch_model_assets.sh: fetch the embedded model
# assets from Hugging Face into crates/cli/assets/, verifying every file
# against the sha256 recorded in crates/cli/assets/MODEL_ASSETS.json.
# Idempotent: valid files are kept. Needs only Windows PowerShell 5.1+ and the
# curl.exe that ships with Windows 10 1803+; no jq.
#
# Override the host with GREPPY_MODEL_ASSET_HF_HOST to fetch from a mirror.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$manifest = Get-Content -Raw 'crates/cli/assets/MODEL_ASSETS.json' | ConvertFrom-Json

$hfHost = $env:GREPPY_MODEL_ASSET_HF_HOST
if (-not $hfHost) { $hfHost = if ($manifest.hf_host) { $manifest.hf_host } else { 'https://huggingface.co' } }
$defaultRev = if ($manifest.revision) { $manifest.revision } else { 'main' }

function Get-Sha256([string]$path) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
}

$status = 0
foreach ($asset in $manifest.assets) {
    # Per-asset revision pins an immutable commit; see fetch_model_assets.sh.
    $rev = if ($asset.revision) { $asset.revision } else { $defaultRev }
    $url = "$hfHost/$($asset.hf_repo)/resolve/$rev/$($asset.hf_file)"
    $dest = $asset.dest
    $want = $asset.sha256

    if (Test-Path -LiteralPath $dest) {
        if ((Get-Sha256 $dest) -eq $want) {
            Write-Output "ok       $dest"
            continue
        }
        Write-Output "refetch  $dest (digest mismatch)"
        Remove-Item -LiteralPath $dest -Force
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
    $tmp = "$dest.download"
    Write-Output "fetch    $url"
    & curl.exe --proto '=https' --tlsv1.2 --location --fail --silent --show-error `
        --retry 5 --retry-connrefused --output $tmp $url
    if ($LASTEXITCODE -ne 0) {
        Write-Error "fetch_model_assets: download failed for $($asset.hf_file) (curl exit $LASTEXITCODE)" -ErrorAction Continue
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        $status = 1
        continue
    }
    $got = Get-Sha256 $tmp
    if ($got -ne $want) {
        Write-Error "fetch_model_assets: digest mismatch for $($asset.hf_file) (got $got, want $want)" -ErrorAction Continue
        Remove-Item -LiteralPath $tmp -Force
        $status = 1
        continue
    }
    Move-Item -LiteralPath $tmp -Destination $dest -Force
    Write-Output "ok       $dest"
}
exit $status
