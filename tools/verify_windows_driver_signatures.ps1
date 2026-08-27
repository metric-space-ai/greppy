#!/usr/bin/env pwsh
# Verify the Microsoft dashboard signature class and emit hash-bound evidence.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DriverPath,
    [Parameter(Mandatory = $true)]
    [string]$CatalogPath,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$HlkOid = '1.3.6.1.4.1.311.10.3.5'
$AttestationOid = '1.3.6.1.4.1.311.10.3.5.1'

function Resolve-RegularFile([string]$Path, [string]$Label) {
    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    $item = Get-Item -LiteralPath $resolved -Force
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "$Label must be a regular file: $resolved"
    }
    return $resolved
}

function Get-SignerEvidence([string]$Path, [string]$Label) {
    $signature = Get-AuthenticodeSignature -LiteralPath $Path
    if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
        throw "$Label Authenticode signature is not valid: $($signature.Status)"
    }
    if ($null -eq $signature.SignerCertificate) {
        throw "$Label has no signer certificate"
    }
    return $signature.SignerCertificate
}

$DriverPath = Resolve-RegularFile $DriverPath 'driver'
$CatalogPath = Resolve-RegularFile $CatalogPath 'catalog'
$OutputPath = [IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $OutputPath) {
    throw "refusing to replace existing signature evidence: $OutputPath"
}

& signtool verify /kp /all /v $DriverPath
if ($LASTEXITCODE -ne 0) { throw 'driver failed kernel-policy signature verification' }
& signtool verify /pa /all /v $CatalogPath
if ($LASTEXITCODE -ne 0) { throw 'catalog failed Authenticode signature verification' }

$driverCertificate = Get-SignerEvidence $DriverPath 'driver'
$catalogCertificate = Get-SignerEvidence $CatalogPath 'catalog'
$ekuExtension = @($catalogCertificate.Extensions | Where-Object { $_.Oid.Value -eq '2.5.29.37' })
if ($ekuExtension.Count -ne 1) {
    throw "catalog signer must contain exactly one Enhanced Key Usage extension; found $($ekuExtension.Count)"
}
$decodedEku = [System.Security.Cryptography.X509Certificates.X509EnhancedKeyUsageExtension]::new(
    $ekuExtension[0],
    $ekuExtension[0].Critical
)
$ekuOids = @($decodedEku.EnhancedKeyUsages | ForEach-Object { $_.Value } | Sort-Object -Unique)
if ($HlkOid -notin $ekuOids) {
    throw "catalog signer lacks Windows Hardware Driver Verification EKU $HlkOid"
}
if ($AttestationOid -in $ekuOids) {
    throw "attestation-signed driver is not release eligible; HLK/dashboard signature is required"
}

$evidence = [ordered]@{
    schema_version = 'greppy.windows-driver-signature-evidence.v1'
    signature_class = 'hlk-dashboard'
    driver_sha256 = (Get-FileHash -LiteralPath $DriverPath -Algorithm SHA256).Hash.ToLowerInvariant()
    catalog_sha256 = (Get-FileHash -LiteralPath $CatalogPath -Algorithm SHA256).Hash.ToLowerInvariant()
    driver_signer_subject = $driverCertificate.Subject
    driver_signer_thumbprint = $driverCertificate.Thumbprint.ToUpperInvariant()
    catalog_signer_subject = $catalogCertificate.Subject
    catalog_signer_thumbprint = $catalogCertificate.Thumbprint.ToUpperInvariant()
    catalog_enhanced_key_usage_oids = $ekuOids
    hardware_driver_verification_oid = $HlkOid
    attestation_oid_absent = $true
}

$parent = Split-Path -Parent $OutputPath
if ($parent) { New-Item -ItemType Directory -Force $parent | Out-Null }
[IO.File]::WriteAllText(
    $OutputPath,
    (($evidence | ConvertTo-Json -Depth 4) + [Environment]::NewLine),
    [Text.UTF8Encoding]::new($false)
)
Write-Output $OutputPath
