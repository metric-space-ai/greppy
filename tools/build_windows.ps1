# Build the greppy CLI from source on Windows x64 with the CUDA backend.
#
# Prerequisites: Rust (rustup honours rust-toolchain.toml), Visual Studio 2022
# Build Tools with the "Desktop development with C++" workload, and the NVIDIA
# CUDA Toolkit (12.8+ for RTX 50-series). This script does what CI gets from
# ilammy/msvc-dev-cmd: it imports the vcvars64 environment so nvcc finds cl.exe,
# puts the toolkit's nvcc on PATH, targets the local GPU's compute capability,
# fetches the pinned model assets, and runs the release build.
#
#   powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1
#   powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1 -CpuOnly
#
# CUDA_ARCH_LIST, when already set, wins over the detected GPU.
param(
    [switch]$CpuOnly
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Import-VsDevEnvironment {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { throw 'vswhere.exe not found: install Visual Studio 2022 Build Tools' }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw 'no Visual Studio install with the MSVC x64 toolset' }
    $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
    # Run vcvars64 in cmd and copy the resulting environment into this process.
    cmd.exe /c "`"$vcvars`" >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
    }
    Write-Output "msvc     $vs"
}

function Resolve-CudaHome {
    # A fresh toolkit install sets CUDA_PATH machine-wide; a shell opened before
    # the install does not see it, so read the registry-backed value too.
    $cuda = $env:CUDA_PATH
    if (-not $cuda) { $cuda = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine') }
    if (-not $cuda) {
        $root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
        $cuda = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
            Sort-Object { [version]($_.Name.TrimStart('v')) } -Descending |
            Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $cuda -or -not (Test-Path (Join-Path $cuda 'bin\nvcc.exe'))) {
        throw 'CUDA Toolkit not found (no nvcc.exe). Install it, e.g. `winget install Nvidia.CUDA --version 12.9`, or pass -CpuOnly.'
    }
    return $cuda
}

& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'fetch_model_assets.ps1')
if ($LASTEXITCODE -ne 0) { throw "fetch_model_assets.ps1 failed (exit $LASTEXITCODE)" }

$cargoArgs = @('build', '--locked', '--release', '--bin', 'greppy')
if ($CpuOnly) {
    $cargoArgs += @('--features', 'cpu-only')
} else {
    Import-VsDevEnvironment
    $cudaHome = Resolve-CudaHome
    $env:CUDA_PATH = $cudaHome
    $env:CUDA_HOME = $cudaHome
    $env:PATH = "$cudaHome\bin;$env:PATH"
    Write-Output "cuda     $cudaHome"
    if (-not $env:CUDA_ARCH_LIST) {
        $cap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
        if ($cap) { $env:CUDA_ARCH_LIST = $cap.Trim() }
    }
    if ($env:CUDA_ARCH_LIST) { Write-Output "arch     $env:CUDA_ARCH_LIST" }
}

& cargo @cargoArgs
if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit $LASTEXITCODE)" }
Write-Output "built    $repo\target\release\greppy.exe"
