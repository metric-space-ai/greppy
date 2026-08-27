#!/usr/bin/env pwsh
# Build Greppy's per-machine x86_64 MSI with the private workspace driver.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DistDir,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [string]$UnsignedDriverPath,
    [string]$DriverCatalogPath,
    [string]$DriverContractPath,
    [switch]$AllowUnsignedForSmokeTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$WixVersion = '5.0.2'
$WixPackageSha256 = 'f30ef0c74e2a986126539c5780be93ac24e8136eaf723b1937b26272703ae173'
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-FullPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
}

$DistDir = Resolve-FullPath $DistDir
$OutputPath = Resolve-FullPath $OutputPath
$WxsPath = Join-Path $RepoRoot 'platform\windows\Greppy.wxs'
$ForkManifest = Join-Path $RepoRoot 'third_party\winfsp-greppy\upstream.json'
if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw "MSI version must be plain x.y.z: $Version" }

$required = @(
    'greppy.exe',
    'greppy-workspace-provider.exe',
    'greppyworkspacefsp-x64.dll',
    'greppyworkspacefsp-x64.sys',
    'README.md',
    'LICENSE',
    'THIRD_PARTY.md',
    'SECURITY.md',
    'SUPPORT.md',
    'CHANGELOG.md',
    'Cargo.lock',
    'windows_driver_contract.py',
    'verify_windows_driver_signatures.ps1',
    'licenses\WINFSP-GPL-3.0-WITH-FLOSS-EXCEPTION.txt'
)
foreach ($relative in $required) {
    $candidate = Join-Path $DistDir $relative
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "MSI staging directory is missing $relative"
    }
}

$reported = (& (Join-Path $DistDir 'greppy.exe') --version) -join ' '
if ($LASTEXITCODE -ne 0 -or $reported.Split(' ')[-1] -ne $Version) {
    throw "staged greppy.exe version '$reported' does not match $Version"
}

$signedDriver = Join-Path $DistDir 'greppyworkspacefsp-x64.sys'
if ($AllowUnsignedForSmokeTest) {
    if ($UnsignedDriverPath -or $DriverCatalogPath -or $DriverContractPath) {
        throw 'unsigned smoke builds must not accept signed driver evidence'
    }
    Write-Warning 'building an unsigned MSI for compile/ICE smoke only; it is not release eligible'
} else {
    if (-not $UnsignedDriverPath -or -not $DriverCatalogPath -or -not $DriverContractPath) {
        throw 'release MSI requires -UnsignedDriverPath, -DriverCatalogPath and -DriverContractPath'
    }
    $UnsignedDriverPath = Resolve-FullPath $UnsignedDriverPath
    $DriverCatalogPath = Resolve-FullPath $DriverCatalogPath
    $DriverContractPath = Resolve-FullPath $DriverContractPath
    foreach ($releaseEvidence in @(
        $DriverCatalogPath,
        $DriverContractPath,
        (Join-Path $DistDir 'greppy-windows-driver-signature-evidence.json')
    )) {
        if (-not (Test-Path -LiteralPath $releaseEvidence -PathType Leaf)) {
            throw "release MSI is missing signed driver evidence: $releaseEvidence"
        }
    }
    $signatureEvidence = Join-Path ([IO.Path]::GetTempPath()) "greppy-driver-signature-$([Guid]::NewGuid().ToString('N')).json"
    try {
        & (Join-Path $RepoRoot 'tools\verify_windows_driver_signatures.ps1') `
            -DriverPath $signedDriver `
            -CatalogPath $DriverCatalogPath `
            -OutputPath $signatureEvidence
        if ($LASTEXITCODE -ne 0) { throw 'private workspace driver is not HLK/dashboard signed' }
        python (Join-Path $RepoRoot 'tools\windows_driver_contract.py') verify `
            --unsigned $UnsignedDriverPath `
            --signed $signedDriver `
            --catalog $DriverCatalogPath `
            --fork-manifest $ForkManifest `
            --signature-evidence $signatureEvidence `
            --manifest $DriverContractPath
        if ($LASTEXITCODE -ne 0) { throw 'private workspace driver contract verification failed' }
    }
    finally {
        Remove-Item -LiteralPath $signatureEvidence -Force -ErrorAction SilentlyContinue
    }
}

$scratch = Join-Path ([IO.Path]::GetTempPath()) "greppy-msi-$([Guid]::NewGuid().ToString('N'))"
$toolDir = Join-Path $scratch 'tool'
$sourceDir = Join-Path $scratch 'source'
New-Item -ItemType Directory -Force $toolDir,$sourceDir,(Split-Path -Parent $OutputPath) | Out-Null
try {
    $package = Join-Path $sourceDir "wix.$WixVersion.nupkg"
    Invoke-WebRequest `
        -Uri "https://api.nuget.org/v3-flatcontainer/wix/$WixVersion/wix.$WixVersion.nupkg" `
        -OutFile $package
    $actualPackageSha = (Get-FileHash -Algorithm SHA256 $package).Hash.ToLowerInvariant()
    if ($actualPackageSha -ne $WixPackageSha256) {
        throw "WiX package checksum mismatch: $actualPackageSha"
    }
    $nugetConfig = Join-Path $scratch 'NuGet.Config'
    [IO.File]::WriteAllText(
        $nugetConfig,
        "<?xml version=`"1.0`" encoding=`"utf-8`"?><configuration><packageSources><clear/><add key=`"pinned`" value=`"$sourceDir`"/></packageSources></configuration>",
        [Text.UTF8Encoding]::new($false)
    )
    dotnet tool install wix `
        --tool-path $toolDir `
        --version $WixVersion `
        --configfile $nugetConfig `
        --no-cache `
        --ignore-failed-sources
    if ($LASTEXITCODE -ne 0) { throw 'pinned WiX tool installation failed' }
    $wix = Join-Path $toolDir 'wix.exe'
    $actualVersion = (& $wix --version) -join ' '
    if ($LASTEXITCODE -ne 0 -or -not $actualVersion.StartsWith($WixVersion)) {
        throw "unexpected WiX version: $actualVersion"
    }

    $pdb = [IO.Path]::ChangeExtension($OutputPath, '.wixpdb')
    & $wix build $WxsPath `
        -arch x64 `
        -d "Version=$Version" `
        -d "DistDir=$DistDir" `
        -out $OutputPath `
        -pdb $pdb
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
        throw 'WiX failed to produce the MSI'
    }
    & $wix msi validate $OutputPath -pdb $pdb
    if ($LASTEXITCODE -ne 0) { throw 'MSI ICE validation failed' }

    $installer = New-Object -ComObject WindowsInstaller.Installer
    $database = $installer.OpenDatabase($OutputPath, 0)
    foreach ($entry in @(@('ProductName','Greppy'), @('ProductVersion',$Version), @('Manufacturer','Metric Space AI'))) {
        $view = $database.OpenView("SELECT ``Value`` FROM ``Property`` WHERE ``Property``='$($entry[0])'")
        $view.Execute()
        $record = $view.Fetch()
        if (-not $record -or $record.StringData(1) -ne $entry[1]) {
            throw "MSI property $($entry[0]) is not '$($entry[1])'"
        }
    }
    $hash = (Get-FileHash -Algorithm SHA256 $OutputPath).Hash.ToLowerInvariant()
    Write-Host "packed: $OutputPath"
    Write-Host "sha256: $hash"
    Write-Host "wix: $actualVersion ($WixPackageSha256)"
}
finally {
    Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
}
