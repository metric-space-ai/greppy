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
$nuget = (Get-Command nuget.exe -ErrorAction Stop).Source
$packageParent = Split-Path -Parent $Destination
$packageSpecs = @(
    [pscustomobject]@{
        Id = 'Microsoft.Windows.WDK.x64'
        Sha256 = 'c393d03dfb640b5c92f546b32f6770ef68cd3aaf691956e7d66d8e2c28a1b55e'
        Props = 'build\native\Microsoft.Windows.WDK.x64.props'
    },
    [pscustomobject]@{
        Id = 'Microsoft.Windows.SDK.CPP.x64'
        Sha256 = 'c29ce7a4641cb37ee32ebb8078cc65cfbabc7025076bcfba869039204b1e960d'
        Props = 'build\native\Microsoft.Windows.SDK.cpp.x64.props'
    },
    [pscustomobject]@{
        Id = 'Microsoft.Windows.SDK.CPP'
        Sha256 = '5d31b38205bdd9ac761b4cb39fbbc6b7209b01c11194324afc674d7d119483a0'
        Props = 'build\native\Microsoft.Windows.SDK.cpp.props'
    }
)
$packageRoots = @{}
$propsImports = @()
foreach ($spec in $packageSpecs) {
    $lowerId = $spec.Id.ToLowerInvariant()
    $package = Join-Path $packageParent "$lowerId.$wdkVersion.nupkg"
    $packageRoot = Join-Path $packageParent "$lowerId.$wdkVersion"
    if ((Test-Path -LiteralPath $package) -or (Test-Path -LiteralPath $packageRoot)) {
        throw "refusing to reuse an existing Windows kit package or extraction directory: $lowerId"
    }
    $uri = "https://api.nuget.org/v3-flatcontainer/$lowerId/$wdkVersion/$lowerId.$wdkVersion.nupkg"
    Invoke-WebRequest -Uri $uri -OutFile $package
    $actualHash = (Get-FileHash -LiteralPath $package -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $spec.Sha256) {
        throw "$($spec.Id) NuGet package checksum mismatch: $actualHash"
    }
    & $nuget verify -Signatures $package
    if ($LASTEXITCODE -ne 0) { throw "$($spec.Id) NuGet signature verification failed" }
    [System.IO.Compression.ZipFile]::ExtractToDirectory($package, $packageRoot)
    $propsPath = Join-Path $packageRoot $spec.Props
    if (-not (Test-Path -LiteralPath $propsPath -PathType Leaf)) {
        throw "$($spec.Id) NuGet package lacks required MSBuild props: $propsPath"
    }
    $packageRoots[$spec.Id] = $packageRoot
    $propsImports += $propsPath
}

$wdkRoot = $packageRoots['Microsoft.Windows.WDK.x64']
$wdkContentRoot = Join-Path $wdkRoot 'c'
if ($wdkContentRoot -match '\s') {
    throw "WDK extraction path must not contain whitespace: $wdkContentRoot"
}
$ntifs = Join-Path $wdkContentRoot "Include\$sdkVersion\km\ntifs.h"
if (-not (Test-Path -LiteralPath $ntifs -PathType Leaf)) {
    throw "pinned WDK package lacks ntifs.h: $ntifs"
}
$importLines = foreach ($propsPath in $propsImports) {
    $escaped = [Security.SecurityElement]::Escape($propsPath)
    "  <Import Project=`"$escaped`" Condition=`"Exists('$escaped')`" />"
}
$directoryBuildProps = @('<Project>') + $importLines + @('</Project>', '')
[IO.File]::WriteAllText(
    (Join-Path $Destination 'Directory.Build.props'),
    ($directoryBuildProps -join [Environment]::NewLine),
    [Text.UTF8Encoding]::new($false)
)
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
        '/p:MyNtddiVersion=0x0A000006', '/p:MyWin32Version=0x0A00',
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
