#!/usr/bin/env pwsh
# Build an UNSIGNED .msix package of greppy for Microsoft Store submission.
#
# The Microsoft Store signs Store submissions itself, so this script never
# signs anything. For LOCAL INSTALL TESTING ONLY you can self-sign the output
# (the self-signed cert Subject must equal the manifest Publisher exactly):
#
#   New-SelfSignedCertificate -Type Custom -Subject 'CN=<publisher-from-identity.json>' `
#     -KeyUsage DigitalSignature -FriendlyName 'greppy msix test' `
#     -CertStoreLocation Cert:\CurrentUser\My `
#     -TextExtension @('2.5.29.37={text}1.3.6.1.5.5.7.3.3', '2.5.29.19={text}')
#   # export to PFX, then:
#   signtool sign /fd SHA256 /f greppy-test.pfx /p <password> greppy-<ver>-<arch>.msix
#   # trust the cert (Local Machine -> Trusted People), then:
#   Add-AppxPackage greppy-<ver>-<arch>.msix
#
# NEVER upload a self-signed package to Partner Center; upload the unsigned
# .msix produced by this script.
#
# Requires a Windows host with the Windows 10/11 SDK (makeappx.exe). Run from
# anywhere; paths resolve relative to the repo root.

[CmdletBinding()]
param(
    # Target architecture of the packaged greppy.exe.
    [ValidateSet('x64', 'arm64')]
    [string]$Arch = 'x64',

    # Path to the release greppy.exe (default: cargo release output).
    [string]$BinaryPath = 'target/release/greppy.exe',

    # Package Identity config (see packaging/msix/identity.json).
    [string]$IdentityJson = 'packaging/msix/identity.json',

    # Manifest template.
    [string]$ManifestTemplate = 'packaging/msix/AppxManifest.xml.in',

    # Tile/logo assets directory.
    [string]$AssetsDir = 'packaging/msix/assets',

    # Where to write greppy-<version>-<arch>.msix.
    [string]$OutputDir = '.',

    # Override the version (default: parsed from Cargo.toml [workspace.package]).
    [string]$Version,

    # Smoke-test escape hatch: build with obviously-fake identity values even
    # if identity.json still contains TODO placeholders. The result can NOT be
    # submitted to the Store.
    [switch]$AllowPlaceholderIdentity
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
function Resolve-RepoPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

$BinaryPath = Resolve-RepoPath $BinaryPath
$IdentityJson = Resolve-RepoPath $IdentityJson
$ManifestTemplate = Resolve-RepoPath $ManifestTemplate
$AssetsDir = Resolve-RepoPath $AssetsDir
$OutputDir = Resolve-RepoPath $OutputDir

foreach ($required in $BinaryPath, $IdentityJson, $ManifestTemplate, $AssetsDir) {
    if (-not (Test-Path $required)) { throw "missing required input: $required" }
}

# --- Version: Cargo workspace x.y.z -> MSIX x.y.z.0 -------------------------
if (-not $Version) {
    $inWorkspacePackage = $false
    foreach ($line in Get-Content (Join-Path $RepoRoot 'Cargo.toml')) {
        if ($line -match '^\s*\[workspace\.package\]\s*$') { $inWorkspacePackage = $true; continue }
        if ($line -match '^\s*\[') { $inWorkspacePackage = $false; continue }
        if ($inWorkspacePackage -and $line -match '^\s*version\s*=\s*"([^"]+)"') {
            $Version = $Matches[1]
            break
        }
    }
    if (-not $Version) { throw 'could not parse version from Cargo.toml [workspace.package]' }
}
if ($Version -notmatch '^(\d+)\.(\d+)\.(\d+)$') {
    throw "version '$Version' is not plain x.y.z - MSIX/Store versions cannot carry pre-release suffixes"
}
foreach ($part in $Matches[1], $Matches[2], $Matches[3]) {
    if ([int]$part -gt 65535) { throw "version component $part exceeds the MSIX 16-bit limit" }
}
# The Store requires the 4th (revision) part to be 0; it is reserved by the
# Store: https://learn.microsoft.com/en-us/windows/msix/package/app-package-requirements
$MsixVersion = "$Version.0"

# --- Identity ---------------------------------------------------------------
$identity = Get-Content $IdentityJson -Raw | ConvertFrom-Json
$identityName = [string]$identity.identity_name
$publisher = [string]$identity.publisher
$publisherDisplay = [string]$identity.publisher_display_name

$placeholders = @(@($identityName, $publisher, $publisherDisplay) | Where-Object { $_ -like 'TODO*' -or -not $_ })
if ($placeholders.Count -gt 0) {
    if (-not $AllowPlaceholderIdentity) {
        throw ("packaging/msix/identity.json still contains TODO placeholder identity values. " +
            "Fill identity_name/publisher/publisher_display_name from Partner Center -> Product identity " +
            "(Store product $($identity.store_product_id)), or pass -AllowPlaceholderIdentity for a " +
            "smoke-test build that can NOT be submitted to the Store.")
    }
    Write-Warning 'building with PLACEHOLDER identity - smoke-test only, not submittable to the Store'
    $identityName = 'Placeholder.Greppy.SmokeTest'
    $publisher = 'CN=00000000-0000-0000-0000-000000000000'
    $publisherDisplay = 'PLACEHOLDER (smoke test only)'
}
if ($publisher -notmatch '^CN=') {
    throw "identity publisher must be the exact Partner Center value starting with 'CN=' (got '$publisher')"
}

# --- Layout -----------------------------------------------------------------
$layout = Join-Path ([IO.Path]::GetTempPath()) "greppy-msix-layout-$Arch-$([Guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force (Join-Path $layout 'Assets') | Out-Null
try {
    Copy-Item $BinaryPath (Join-Path $layout 'greppy.exe')
    Copy-Item (Join-Path $AssetsDir '*.png') (Join-Path $layout 'Assets')
    # Ship the model/third-party license notices inside the package, matching
    # the direct-download archives (greppy.exe embeds the models).
    Copy-Item (Join-Path $RepoRoot 'licenses') (Join-Path $layout 'licenses') -Recurse
    Copy-Item (Join-Path $RepoRoot 'LICENSE') (Join-Path $layout 'LICENSE')
    Copy-Item (Join-Path $RepoRoot 'THIRD_PARTY.md') (Join-Path $layout 'THIRD_PARTY.md')

    $manifest = Get-Content $ManifestTemplate -Raw
    foreach ($pair in @(
            @('@IDENTITY_NAME@', $identityName),
            @('@PUBLISHER@', $publisher),
            @('@PUBLISHER_DISPLAY@', $publisherDisplay),
            @('@VERSION@', $MsixVersion),
            @('@ARCH@', $Arch))) {
        $manifest = $manifest.Replace($pair[0], $pair[1])
    }
    $left = [regex]::Matches($manifest, '@[A-Z_]+@') | ForEach-Object { $_.Value } | Select-Object -Unique
    if ($left) { throw "unsubstituted manifest placeholders: $($left -join ', ')" }
    # makeappx requires UTF-8; avoid the BOM some Set-Content encodings add.
    [IO.File]::WriteAllText((Join-Path $layout 'AppxManifest.xml'), $manifest, [Text.UTF8Encoding]::new($false))

    # --- makeappx pack ------------------------------------------------------
    $makeappxCmd = Get-Command makeappx.exe -ErrorAction SilentlyContinue
    $makeappx = if ($makeappxCmd) { $makeappxCmd.Source } else { $null }
    if (-not $makeappx) {
        $kitsBin = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
        $hostArch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'x64' }
        $candidates = @()
        foreach ($binArch in @($hostArch, 'x64') | Select-Object -Unique) {
            $candidates += Get-ChildItem -Path (Join-Path $kitsBin "10.*\$binArch\makeappx.exe") -ErrorAction SilentlyContinue
        }
        $makeappx = $candidates | Sort-Object { [Version]$_.Directory.Parent.Name } -Descending |
            Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $makeappx) { throw 'makeappx.exe not found - install the Windows 10/11 SDK' }
    Write-Host "using makeappx: $makeappx"

    New-Item -ItemType Directory -Force $OutputDir | Out-Null
    $msixPath = Join-Path $OutputDir "greppy-$Version-$Arch.msix"
    if (Test-Path $msixPath) { Remove-Item $msixPath }
    & $makeappx pack /o /h SHA256 /d $layout /p $msixPath
    if ($LASTEXITCODE -ne 0) { throw "makeappx pack failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path $msixPath)) { throw "makeappx reported success but $msixPath is missing" }

    # --- Validate the packed manifest ---------------------------------------
    # Get-AppxPackageManifest only works on INSTALLED packages, so read the
    # manifest straight out of the .msix (it is a zip) and assert the fields
    # the Store validates.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($msixPath)
    try {
        $entryNames = $zip.Entries | ForEach-Object { $_.FullName }
        foreach ($must in 'AppxManifest.xml', 'greppy.exe', 'Assets/StoreLogo.png',
            'Assets/Square44x44Logo.png', 'Assets/Square150x150Logo.png') {
            if ($entryNames -notcontains $must) { throw "packed msix is missing $must" }
        }
        $entry = $zip.GetEntry('AppxManifest.xml')
        $reader = [IO.StreamReader]::new($entry.Open())
        try { [xml]$packed = $reader.ReadToEnd() } finally { $reader.Dispose() }

        $id = $packed.Package.Identity
        if ($id.Name -ne $identityName) { throw "packed Identity/Name '$($id.Name)' != '$identityName'" }
        if ($id.Publisher -ne $publisher) { throw "packed Identity/Publisher '$($id.Publisher)' != '$publisher'" }
        if ($id.Version -ne $MsixVersion) { throw "packed Identity/Version '$($id.Version)' != '$MsixVersion'" }
        if ($id.ProcessorArchitecture -ne $Arch) { throw "packed ProcessorArchitecture '$($id.ProcessorArchitecture)' != '$Arch'" }

        # XPath by local-name(): the elements are namespace-qualified
        # (uap5:ExecutionAlias, rescap:Capability).
        $aliases = @($packed.SelectNodes("//*[local-name()='ExecutionAlias']") | ForEach-Object { $_.Alias })
        if ($aliases -notcontains 'greppy.exe') { throw 'packed manifest lost the greppy.exe AppExecutionAlias' }

        $caps = @($packed.SelectNodes("//*[local-name()='Capability']") | ForEach-Object { $_.Name })
        if ($caps -notcontains 'runFullTrust') { throw 'packed manifest lost the runFullTrust capability' }
    }
    finally { $zip.Dispose() }

    # The packaged binary must report the same version the manifest claims.
    $reported = (& $BinaryPath --version) -join ' '
    if ($LASTEXITCODE -ne 0) { throw 'greppy.exe --version failed' }
    if ($reported.Split(' ')[-1] -ne $Version) {
        throw "greppy.exe reports '$reported' but the package version is $Version"
    }

    $sha256 = (Get-FileHash -Algorithm SHA256 $msixPath).Hash.ToLowerInvariant()
    Write-Host "packed:  $msixPath"
    Write-Host "sha256:  $sha256"
    Write-Host "version: $MsixVersion  arch: $Arch  identity: $identityName"
    Write-Host 'NOTE: package is UNSIGNED by design - the Microsoft Store signs the submission.'
}
finally {
    Remove-Item -Recurse -Force $layout -ErrorAction SilentlyContinue
}
