param(
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if (Test-Path -LiteralPath $Destination) {
    throw "destination already exists: $Destination"
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$forkRoot = Join-Path $repositoryRoot 'third_party/winfsp-greppy'
$manifestPath = Join-Path $forkRoot 'upstream.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

& python (Join-Path $repositoryRoot 'tools/verify_winfsp_fork.py') $forkRoot
if ($LASTEXITCODE -ne 0) {
    throw 'WinFsp fork metadata verification failed'
}

& git init --quiet $Destination
if ($LASTEXITCODE -ne 0) { throw 'cannot initialize WinFsp source directory' }
& git -C $Destination remote add origin $manifest.repository
if ($LASTEXITCODE -ne 0) { throw 'cannot add WinFsp upstream remote' }
& git -C $Destination fetch --quiet --depth 1 origin $manifest.commit
if ($LASTEXITCODE -ne 0) { throw 'cannot fetch exact WinFsp upstream commit' }
& git -C $Destination checkout --quiet --detach FETCH_HEAD
if ($LASTEXITCODE -ne 0) { throw 'cannot check out exact WinFsp upstream commit' }
$actualCommit = (& git -C $Destination rev-parse HEAD).Trim()
if ($actualCommit -ne $manifest.commit) {
    throw "WinFsp source commit mismatch: $actualCommit"
}

foreach ($patch in $manifest.patches) {
    $patchPath = Join-Path $forkRoot $patch.path
    & git -C $Destination apply --check $patchPath
    if ($LASTEXITCODE -ne 0) { throw "WinFsp patch does not apply: $($patch.path)" }
    & git -C $Destination apply $patchPath
    if ($LASTEXITCODE -ne 0) { throw "cannot apply WinFsp patch: $($patch.path)" }
}

$expectedFiles = @($manifest.patches.modified_files | Sort-Object -Unique)
$actualFiles = @(& git -C $Destination diff --name-only | Sort-Object -Unique)
$difference = @(Compare-Object -ReferenceObject $expectedFiles -DifferenceObject $actualFiles)
if ($difference.Count -ne 0) {
    throw "patched WinFsp file set differs from manifest: $($difference | Out-String)"
}
& git -C $Destination diff --check
if ($LASTEXITCODE -ne 0) { throw 'patched WinFsp source fails git diff --check' }

$sdkVersion = '10.0.19041.0'
$sdkInclude = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\Include\$sdkVersion"
if (-not (Test-Path -LiteralPath $sdkInclude -PathType Container)) {
    throw "required Windows SDK is absent: $sdkInclude"
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw "vswhere is absent: $vswhere"
}
$visualStudio = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if (-not $visualStudio) { throw 'Visual Studio C++ toolchain is absent' }
$developerShell = Join-Path $visualStudio 'Common7\Tools\VsDevCmd.bat'
if (-not (Test-Path -LiteralPath $developerShell -PathType Leaf)) {
    throw "Visual Studio developer shell is absent: $developerShell"
}

$solutionRoot = Join-Path $Destination 'build\VStudio'
$projects = @(
    'winfsp_sys.vcxproj',
    'winfsp_dll.vcxproj',
    'testing\winfsp-tests.vcxproj'
)
foreach ($project in $projects) {
    $projectPath = Join-Path $solutionRoot $project
    $arguments = @(
        'call', ('"' + $developerShell + '"'), '-arch=x64', '-host_arch=x64', '>', 'nul', '&&',
        'msbuild', ('"' + $projectPath + '"'), '/m', '/nologo', '/verbosity:minimal',
        '/p:Configuration=Release', '/p:Platform=x64', "/p:MyTargetPlatformVersion=$sdkVersion",
        '/p:MyNtddiVersion=0x0A000006', '/p:MyWin32Version=0x0A00'
    )
    & cmd.exe /D /S /C ($arguments -join ' ')
    if ($LASTEXITCODE -ne 0) { throw "WinFsp project build failed: $project" }
}

$outputRoot = Join-Path $solutionRoot 'build\Release'
$requiredOutputs = @(
    'winfsp-x64.sys',
    'winfsp-x64.dll',
    'winfsp-tests-x64.exe'
)
foreach ($output in $requiredOutputs) {
    $path = Join-Path $outputRoot $output
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "required WinFsp build output is absent: $path"
    }
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Output "$hash  $path"
}
