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

$wdkVersion = '10.0.26100.6584'
$sdkVersion = '10.0.26100.0'
$wdkPackageSha256 = 'c393d03dfb640b5c92f546b32f6770ef68cd3aaf691956e7d66d8e2c28a1b55e'
$wdkPackage = Join-Path (Split-Path -Parent $Destination) "microsoft.windows.wdk.x64.$wdkVersion.nupkg"
$wdkRoot = Join-Path (Split-Path -Parent $Destination) "microsoft.windows.wdk.x64.$wdkVersion"
if ((Test-Path -LiteralPath $wdkPackage) -or (Test-Path -LiteralPath $wdkRoot)) {
    throw 'refusing to reuse an existing WDK package or extraction directory'
}
$wdkUri = "https://api.nuget.org/v3-flatcontainer/microsoft.windows.wdk.x64/$wdkVersion/microsoft.windows.wdk.x64.$wdkVersion.nupkg"
Invoke-WebRequest -Uri $wdkUri -OutFile $wdkPackage
$actualWdkHash = (Get-FileHash -LiteralPath $wdkPackage -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualWdkHash -ne $wdkPackageSha256) {
    throw "WDK NuGet package checksum mismatch: $actualWdkHash"
}
$nuget = (Get-Command nuget.exe -ErrorAction Stop).Source
& $nuget verify -Signatures $wdkPackage
if ($LASTEXITCODE -ne 0) { throw 'WDK NuGet signature verification failed' }
[System.IO.Compression.ZipFile]::ExtractToDirectory($wdkPackage, $wdkRoot)
$wdkContentRoot = Join-Path $wdkRoot 'c'
if ($wdkContentRoot -match '\s') {
    throw "WDK extraction path must not contain whitespace: $wdkContentRoot"
}
$ntifs = Join-Path $wdkContentRoot "Include\$sdkVersion\km\ntifs.h"
if (-not (Test-Path -LiteralPath $ntifs -PathType Leaf)) {
    throw "pinned WDK package lacks ntifs.h: $ntifs"
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
        'set', ('"WindowsSdkDir=' + $wdkContentRoot + '\"'), '&&',
        'set', ('"WindowsSdkDir_10=' + $wdkContentRoot + '\"'), '&&',
        'set', ('"WDKContentRoot=' + $wdkContentRoot + '\"'), '&&',
        'set', ('"UCRTContentRoot=' + $wdkContentRoot + '\"'), '&&',
        'set', '"WDK_NuGet=true"', '&&',
        'msbuild', ('"' + $projectPath + '"'), '/m', '/nologo', '/verbosity:minimal',
        '/p:Configuration=Release', '/p:Platform=x64', "/p:MyTargetPlatformVersion=$sdkVersion",
        '/p:MyNtddiVersion=0x0A000006', '/p:MyWin32Version=0x0A00',
        ('/p:WindowsSdkDir=' + $wdkContentRoot + '\'),
        ('/p:WindowsSdkDir_10=' + $wdkContentRoot + '\'),
        ('/p:WDKContentRoot=' + $wdkContentRoot + '\'),
        ('/p:UCRTContentRoot=' + $wdkContentRoot + '\'),
        '/p:WDK_NuGet=true'
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
