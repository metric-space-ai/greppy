#!/usr/bin/env pwsh
# End-to-end inference-daemon acceptance for an installed Windows release.
# It drives only the public CLI and newline-framed named-pipe protocol.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Binary,
    [string]$Work = (Join-Path ([IO.Path]::GetTempPath()) "greppy-daemon-stress-$([Guid]::NewGuid().ToString('N'))"),
    [ValidateRange(30, 3600)]
    [int]$ChildTimeoutSeconds = 900
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Binary = [IO.Path]::GetFullPath($Binary)
$Work = [IO.Path]::GetFullPath($Work)
if (-not (Test-Path -LiteralPath $Binary -PathType Leaf)) {
    throw "greppy binary is not a regular file: $Binary"
}
New-Item -ItemType Directory -Force $Work,(Join-Path $Work 'store'),(Join-Path $Work 'runtime'),(Join-Path $Work 'out') | Out-Null

$env:GREPPY_STORE_DIR = Join-Path $Work 'store'
$env:GREPPY_RUNTIME_DIR = Join-Path $Work 'runtime'
$env:GREPPY_DEVICE = 'cpu'
$env:GREPPY_EMBED_DAEMON_MODEL_TTL_S = '600'
$env:GREPPY_EMBED_DAEMON_EXIT_TTL_S = '600'
$env:GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S = '5'
$env:GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S = '15'

Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO.Pipes;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

public sealed class GreppyPipeResult
{
    public string RequestId { get; set; }
    public string Response { get; set; }
    public string Error { get; set; }
    public long ElapsedMilliseconds { get; set; }
}

public static class GreppyPipeClient
{
    public static string PipeName(string endpoint)
    {
        const string Prefix = @"\\.\pipe\";
        if (String.IsNullOrEmpty(endpoint) ||
            !endpoint.StartsWith(Prefix, StringComparison.OrdinalIgnoreCase) ||
            endpoint.Length == Prefix.Length)
            throw new ArgumentException("not a local Windows named-pipe endpoint");
        return endpoint.Substring(Prefix.Length);
    }

    private static string JsonString(string value)
    {
        StringBuilder encoded = new StringBuilder("\"");
        foreach (char character in value)
        {
            switch (character)
            {
                case '\"': encoded.Append("\\\""); break;
                case '\\': encoded.Append("\\\\"); break;
                case '\b': encoded.Append("\\b"); break;
                case '\f': encoded.Append("\\f"); break;
                case '\n': encoded.Append("\\n"); break;
                case '\r': encoded.Append("\\r"); break;
                case '\t': encoded.Append("\\t"); break;
                default:
                    if (character < 0x20)
                        encoded.Append("\\u").Append(((int)character).ToString("x4"));
                    else
                        encoded.Append(character);
                    break;
            }
        }
        return encoded.Append('\"').ToString();
    }

    public static GreppyPipeResult RoundTripText(
        string endpoint, string requestId, string text, int timeoutMilliseconds)
    {
        return RoundTripBytes(
            endpoint,
            requestId,
            Encoding.UTF8.GetBytes(text.EndsWith("\n", StringComparison.Ordinal)
                ? text
                : text + "\n"),
            timeoutMilliseconds);
    }

    public static GreppyPipeResult RoundTripBytes(
        string endpoint, string requestId, byte[] payload, int timeoutMilliseconds)
    {
        Stopwatch elapsed = Stopwatch.StartNew();
        try
        {
            using (CancellationTokenSource cancellation =
                new CancellationTokenSource(timeoutMilliseconds))
            using (NamedPipeClientStream pipe = new NamedPipeClientStream(
                ".", PipeName(endpoint), PipeDirection.InOut,
                PipeOptions.Asynchronous))
            {
                pipe.ConnectAsync(cancellation.Token).GetAwaiter().GetResult();
                pipe.WriteAsync(payload, 0, payload.Length, cancellation.Token)
                    .GetAwaiter().GetResult();
                pipe.FlushAsync(cancellation.Token).GetAwaiter().GetResult();
                List<byte> response = new List<byte>();
                byte[] buffer = new byte[65536];
                while (true)
                {
                    int count = pipe.ReadAsync(
                        buffer, 0, buffer.Length, cancellation.Token)
                        .GetAwaiter().GetResult();
                    if (count == 0)
                        break;
                    for (int index = 0; index < count; index++)
                    {
                        if (buffer[index] == (byte)'\n')
                            return new GreppyPipeResult {
                                RequestId = requestId,
                                Response = Encoding.UTF8.GetString(response.ToArray()),
                                ElapsedMilliseconds = elapsed.ElapsedMilliseconds
                            };
                        response.Add(buffer[index]);
                    }
                }
                return new GreppyPipeResult {
                    RequestId = requestId,
                    Error = "connection closed without a complete response",
                    ElapsedMilliseconds = elapsed.ElapsedMilliseconds
                };
            }
        }
        catch (Exception error)
        {
            return new GreppyPipeResult {
                RequestId = requestId,
                Error = error.GetType().Name + ": " + error.Message,
                ElapsedMilliseconds = elapsed.ElapsedMilliseconds
            };
        }
    }

    public static Task<GreppyPipeResult> BeginRaw(
        string endpoint, string requestId, byte[] payload, int timeoutMilliseconds)
    {
        return Task.Run(() =>
            RoundTripBytes(endpoint, requestId, payload, timeoutMilliseconds));
    }

    private static GreppyPipeResult[] BurstRequests(
        string endpoint, string prefix, string[] requests,
        int timeoutMilliseconds)
    {
        int count = requests.Length;
        Task<GreppyPipeResult>[] tasks = new Task<GreppyPipeResult>[count];
        for (int index = 0; index < count; index++)
        {
            int requestIndex = index;
            string requestId = prefix + "-" + requestIndex.ToString();
            tasks[requestIndex] = Task.Run(() => {
                GreppyPipeResult result = null;
                for (int attempt = 0; attempt < 3; attempt++)
                {
                    result = RoundTripText(
                        endpoint, requestId, requests[requestIndex], timeoutMilliseconds);
                    if (String.IsNullOrEmpty(result.Error))
                        return result;
                    Thread.Sleep(100 * (attempt + 1));
                }
                return result;
            });
        }
        Task.WaitAll(tasks);
        GreppyPipeResult[] results = new GreppyPipeResult[count];
        for (int index = 0; index < count; index++)
            results[index] = tasks[index].Result;
        return results;
    }

    public static GreppyPipeResult[] BurstEmbedding(
        string endpoint, string prefix, int count, int timeoutMilliseconds,
        string promptVersion, string modelKey)
    {
        string[] requests = new string[count];
        for (int index = 0; index < count; index++)
        {
            string requestId = prefix + "-" + index.ToString();
            requests[index] = "{\"protocol\":2,\"request_id\":" +
                JsonString(requestId) + ",\"pv\":" + JsonString(promptVersion) +
                ",\"mk\":" + JsonString(modelKey) + ",\"text\":" +
                JsonString("capacity flood " + requestId) + "}";
        }
        return BurstRequests(endpoint, prefix, requests, timeoutMilliseconds);
    }

    public static GreppyPipeResult[] BurstSummary(
        string endpoint, string prefix, int count, int timeoutMilliseconds,
        string promptVersion, string filterVersion, string modelKey)
    {
        string[] requests = new string[count];
        for (int index = 0; index < count; index++)
        {
            string requestId = prefix + "-" + index.ToString();
            requests[index] = "{\"protocol\":2,\"request_id\":" +
                JsonString(requestId) + ",\"pv\":" + JsonString(promptVersion) +
                ",\"fv\":" + JsonString(filterVersion) +
                ",\"mk\":" + JsonString(modelKey) +
                ",\"mode\":\"brief\",\"path\":" +
                JsonString("src/flood.rs") + ",\"source\":" +
                JsonString("pub fn capacity_flood() -> usize { 48 }") + "}";
        }
        return BurstRequests(endpoint, prefix, requests, timeoutMilliseconds);
    }
}
'@

$DaemonPids = [Collections.Generic.HashSet[int]]::new()
$ChildProcesses = [Collections.Generic.List[Diagnostics.Process]]::new()
$QuerySequence = 0
$PrewarmNavigation = $null

function Write-Section([string]$Name) {
    Write-Host ""
    Write-Host "=== $Name ==="
}

function Wait-For(
    [string]$Description,
    [int]$TimeoutSeconds,
    [scriptblock]$Condition
) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $lastError = $null
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            if (& $Condition) { return }
        }
        catch {
            $lastError = $_.Exception.Message
        }
        Start-Sleep -Milliseconds 100
    }
    if ($lastError) {
        throw "timed out waiting for '$Description'; last error: $lastError"
    }
    throw "timed out waiting for $Description"
}

function Invoke-PipeJson(
    [string]$Endpoint,
    [hashtable]$Request,
    [int]$TimeoutSeconds = 10
) {
    $requestId = if ($Request.ContainsKey('request_id')) {
        [string]$Request.request_id
    } else {
        ''
    }
    $payload = $Request | ConvertTo-Json -Compress -Depth 8
    $result = [GreppyPipeClient]::RoundTripText(
        $Endpoint, $requestId, $payload, $TimeoutSeconds * 1000)
    if ($result.Error) {
        throw "named-pipe request failed: $($result.Error)"
    }
    return ($result.Response | ConvertFrom-Json)
}

function Get-DaemonStatus([string]$Endpoint) {
    $status = Invoke-PipeJson $Endpoint @{ protocol = 2; op = 'status' } 5
    if ([int]$status.daemon_pid -gt 0) {
        [void]$DaemonPids.Add([int]$status.daemon_pid)
    }
    return $status
}

function Write-FailureDiagnostics {
    Write-Host ""
    Write-Host '=== fail-closed daemon diagnostics ==='
    try {
        $processes = @(
            Get-CimInstance Win32_Process |
                Where-Object { $_.ExecutablePath -eq $Binary } |
                Select-Object ProcessId,CreationDate,CommandLine
        )
        Write-Host ($processes | ConvertTo-Json -Depth 3 -Compress)
    }
    catch {
        Write-Host "process inventory unavailable: $($_.Exception.Message)"
    }
    if ($null -ne $script:PrewarmNavigation) {
        $prewarmDiagnostic = [pscustomobject]@{
            process_id = $script:PrewarmNavigation.Process.Id
            process_exited = $script:PrewarmNavigation.Process.HasExited
            stdout_completed = $script:PrewarmNavigation.Stdout.IsCompleted
            stderr_completed = $script:PrewarmNavigation.Stderr.IsCompleted
            elapsed_ms = [Math]::Round(
                ([DateTime]::UtcNow - $script:PrewarmNavigation.StartedAtUtc).TotalMilliseconds,
                3
            )
        }
        Write-Host "prewarm-navigation=$($prewarmDiagnostic | ConvertTo-Json -Compress)"
    }
    try {
        $runtimeInventory = @(
            Get-ChildItem -LiteralPath $env:GREPPY_RUNTIME_DIR -Force -Recurse |
                Select-Object FullName,Length,LastWriteTimeUtc,Attributes
        )
        Write-Host ($runtimeInventory | ConvertTo-Json -Depth 3 -Compress)
    }
    catch {
        Write-Host "runtime inventory unavailable: $($_.Exception.Message)"
    }
    try {
        $backgroundJobs = @(
            Get-ChildItem -LiteralPath $env:GREPPY_STORE_DIR -Filter index.job -File -Recurse |
                ForEach-Object {
                    [pscustomobject]@{
                        path = $_.FullName
                        content = Get-Content -LiteralPath $_.FullName -Raw
                    }
                }
        )
        Write-Host "background-jobs=$($backgroundJobs | ConvertTo-Json -Depth 4 -Compress)"
    }
    catch {
        Write-Host "background job inventory unavailable: $($_.Exception.Message)"
    }
    try {
        $indexOutputs = @(
            Get-ChildItem -LiteralPath (Join-Path $Work 'out') -Filter 'index-*.txt*' -File |
                ForEach-Object {
                    [pscustomobject]@{
                        path = $_.FullName
                        content = Get-Content -LiteralPath $_.FullName -Raw
                    }
                }
        )
        Write-Host "index-outputs=$($indexOutputs | ConvertTo-Json -Depth 4 -Compress)"
    }
    catch {
        Write-Host "index output inventory unavailable: $($_.Exception.Message)"
    }
    try {
        if ($RepoEmbed) {
            $status = & $Binary --root $RepoEmbed index status --json 2>&1
            Write-Host "index-status=$($status -join [Environment]::NewLine)"
        }
    }
    catch {
        Write-Host "index status unavailable: $($_.Exception.Message)"
    }
    foreach ($endpointName in @('EmbedEndpoint', 'SummaryEndpoint')) {
        $endpointVariable = Get-Variable -Name $endpointName -ErrorAction SilentlyContinue
        if ($null -eq $endpointVariable) { continue }
        try {
            $status = Get-DaemonStatus ([string]$endpointVariable.Value)
            Write-Host "$endpointName=$($status | ConvertTo-Json -Depth 5 -Compress)"
        }
        catch {
            Write-Host "$endpointName unavailable: $($_.Exception.Message)"
        }
    }
}

function Test-ProcessAlive([int]$ProcessId) {
    return $null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

function Stop-Daemon([int]$ProcessId) {
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    Wait-For "daemon process $ProcessId to exit" 15 {
        -not (Test-ProcessAlive $ProcessId)
    }
}

function New-Query([string]$Label) {
    $script:QuerySequence++
    return "windows-stress-$PID-$($script:QuerySequence)-$Label"
}

function Start-GreppyProcess([string[]]$Arguments) {
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $Binary
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($argument in $Arguments) {
        [void]$start.ArgumentList.Add($argument)
    }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $start
    if (-not $process.Start()) {
        throw "failed to start greppy $($Arguments -join ' ')"
    }
    [void]$ChildProcesses.Add($process)
    return [pscustomobject]@{
        Process = $process
        Stdout = $process.StandardOutput.ReadToEndAsync()
        Stderr = $process.StandardError.ReadToEndAsync()
        StartedAtUtc = [DateTime]::UtcNow
        CommandLabel = "greppy $($Arguments -join ' ')"
    }
}

function Wait-GreppyChild(
    [pscustomobject]$Handle,
    [string]$Operation
) {
    $elapsedMilliseconds = ([DateTime]::UtcNow - $Handle.StartedAtUtc).TotalMilliseconds
    $remainingMilliseconds = [Math]::Floor(
        ($ChildTimeoutSeconds * 1000.0) - $elapsedMilliseconds
    )
    if ($remainingMilliseconds -gt 0 -and
        $Handle.Process.WaitForExit([int]$remainingMilliseconds)) {
        return
    }

    try {
        Stop-Process -Id $Handle.Process.Id -Force -ErrorAction SilentlyContinue
        [void]$Handle.Process.WaitForExit(10000)
    }
    catch {
        # The shared finally block retries cleanup for every tracked child.
    }
    throw "$Operation exceeded the $ChildTimeoutSeconds-second child deadline ($($Handle.CommandLabel))"
}

function Complete-GreppyProcess(
    [pscustomobject]$Handle,
    [string]$OutputPath,
    [bool]$RequireSuccess
) {
    Wait-GreppyChild $Handle 'concurrent semantic search'
    $stdout = $Handle.Stdout.GetAwaiter().GetResult()
    $stderr = $Handle.Stderr.GetAwaiter().GetResult()
    [IO.File]::WriteAllText($OutputPath, $stdout, [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText("$OutputPath.stderr", $stderr, [Text.UTF8Encoding]::new($false))
    if ($RequireSuccess -and $Handle.Process.ExitCode -ne 0) {
        throw "greppy child exited $($Handle.Process.ExitCode): $stderr"
    }
    if ($RequireSuccess) {
        $json = $stdout | ConvertFrom-Json
        if ($json.status -ne 'ok' -or @($json.hits).Count -lt 1) {
            throw "greppy child returned no successful search hit: $stdout"
        }
    }
}

function Invoke-GreppyJson(
    [string[]]$Arguments,
    [string]$OutputPath,
    [int[]]$AllowedExitCodes = @(0)
) {
    $process = Start-GreppyProcess $Arguments
    Wait-GreppyChild $process 'Greppy JSON command'
    $stdout = $process.Stdout.GetAwaiter().GetResult()
    $stderr = $process.Stderr.GetAwaiter().GetResult()
    [IO.File]::WriteAllText($OutputPath, $stdout, [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText("$OutputPath.stderr", $stderr, [Text.UTF8Encoding]::new($false))
    if ($AllowedExitCodes -notcontains $process.Process.ExitCode) {
        throw "greppy $($Arguments -join ' ') exited $($process.Process.ExitCode): $stderr"
    }
    if ([String]::IsNullOrWhiteSpace($stdout)) {
        throw "greppy $($Arguments -join ' ') produced no JSON"
    }
    return ($stdout | ConvertFrom-Json)
}

function Invoke-Semantic([string]$Label, [string]$Query) {
    $path = Join-Path $Work "out\$Label.json"
    $result = Invoke-GreppyJson @(
        '--root', $RepoEmbed, 'search', '--json', $Query
    ) $path
    if ($result.status -ne 'ok' -or @($result.hits).Count -lt 1) {
        throw "semantic query returned no successful hit"
    }
    return $result
}

function Get-EmbeddingModelKey(
    [pscustomobject]$SearchResult,
    [string]$StoreRoot
) {
    foreach ($field in @('model_id', 'prompt_version', 'task_profile')) {
        $property = $SearchResult.PSObject.Properties[$field]
        if ($null -eq $property -or [String]::IsNullOrWhiteSpace([string]$property.Value)) {
            throw "semantic result lacks $field required by the daemon protocol"
        }
    }
    $modelRoot = Join-Path $StoreRoot 'models\v1\embeddinggemma-300m-q4k'
    $gguf = @(Get-ChildItem -LiteralPath $modelRoot -Filter '*.gguf' -File -Recurse)
    $tokenizer = @(Get-ChildItem -LiteralPath $modelRoot -Filter 'tokenizer.json' -File -Recurse)
    if ($gguf.Count -ne 1 -or $tokenizer.Count -ne 1) {
        throw "expected one managed EmbeddingGemma GGUF and tokenizer, got $($gguf.Count) and $($tokenizer.Count)"
    }
    $ggufHash = (Get-FileHash -LiteralPath $gguf[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $tokenizerHash = (Get-FileHash -LiteralPath $tokenizer[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    return "$($SearchResult.model_id)|$($SearchResult.prompt_version)|$($SearchResult.task_profile)|gguf;$ggufHash;$tokenizerHash"
}

function Get-SummaryModelContract(
    [pscustomobject]$Doctor,
    [string]$StoreRoot
) {
    $summary = $Doctor.inference.models.summary
    foreach ($field in @('model_id', 'prompt_version')) {
        $property = $summary.PSObject.Properties[$field]
        if ($null -eq $property -or [String]::IsNullOrWhiteSpace([string]$property.Value)) {
            throw "summary model status lacks $field required by the daemon protocol"
        }
    }
    $triagePromptVersion = 'qwen35-triage-v3'
    $filterVersion = 'qwen35-brief-filter-v3'
    $modelRoot = Join-Path $StoreRoot 'models\v1\qwen35-0.8b-mtp-q4km'
    $gguf = @(Get-ChildItem -LiteralPath $modelRoot -Filter '*.gguf' -File -Recurse)
    $tokenizer = @(Get-ChildItem -LiteralPath $modelRoot -Filter 'tokenizer.json' -File -Recurse)
    if ($gguf.Count -ne 1 -or $tokenizer.Count -ne 1) {
        throw "expected one managed Qwen GGUF and tokenizer, got $($gguf.Count) and $($tokenizer.Count)"
    }
    $ggufHash = (Get-FileHash -LiteralPath $gguf[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $tokenizerHash = (Get-FileHash -LiteralPath $tokenizer[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    return [pscustomobject]@{
        prompt_version = [string]$summary.prompt_version
        filter_version = $filterVersion
        model_key = "$($summary.model_id):$($summary.prompt_version):$triagePromptVersion`:$filterVersion`:cpu`:$ggufHash`:$tokenizerHash"
    }
}

function Convert-Burst([GreppyPipeResult[]]$Results) {
    $parsed = @()
    foreach ($result in $Results) {
        if ($result.Error) {
            $parsed += [pscustomobject]@{
                request_id = $result.RequestId
                client_error = $result.Error
                raw_response = $result.Response
                response_error = $null
                error_kind = $null
                retryable = $false
                capacity = $false
                echo_ok = $false
                inference_ok = $false
            }
            continue
        }
        $response = $result.Response | ConvertFrom-Json
        $requestIdProperty = $response.PSObject.Properties['request_id']
        $responseErrorProperty = $response.PSObject.Properties['error']
        $errorKindProperty = $response.PSObject.Properties['error_kind']
        $retryableProperty = $response.PSObject.Properties['retryable']
        $vectorProperty = $response.PSObject.Properties['v_bits']
        $summaryProperty = $response.PSObject.Properties['s']
        $parsed += [pscustomobject]@{
            request_id = $result.RequestId
            client_error = $null
            raw_response = $result.Response
            response_error = if ($null -ne $responseErrorProperty) {
                $responseErrorProperty.Value
            } else { $null }
            error_kind = if ($null -ne $errorKindProperty) {
                $errorKindProperty.Value
            } else { $null }
            retryable = (
                $null -ne $retryableProperty -and
                $retryableProperty.Value -eq $true
            )
            capacity = (
                $null -ne $errorKindProperty -and
                $errorKindProperty.Value -eq 'capacity' -and
                $null -ne $retryableProperty -and
                $retryableProperty.Value -eq $true
            )
            echo_ok = (
                $null -ne $requestIdProperty -and
                $requestIdProperty.Value -eq $result.RequestId
            )
            inference_ok = (
                $null -ne $requestIdProperty -and
                $requestIdProperty.Value -eq $result.RequestId -and
                $null -ne $vectorProperty -and
                @($vectorProperty.Value).Count -gt 0
            ) -or (
                $null -ne $requestIdProperty -and
                $requestIdProperty.Value -eq $result.RequestId -and
                $null -ne $summaryProperty -and
                @($summaryProperty.Value).Count -gt 0
            )
        }
    }
    return @($parsed)
}

try {
    Write-Section 'fixtures and real model indexes'
    $RepoEmbed = Join-Path $Work 'repo-embed'
    $RepoBrief = Join-Path $Work 'repo-brief'
    $fixtureDirectories = @(
        (Join-Path $RepoEmbed 'src'),
        (Join-Path $RepoEmbed '.git'),
        (Join-Path $RepoBrief 'src'),
        (Join-Path $RepoBrief '.git')
    )
    New-Item -ItemType Directory -Force $fixtureDirectories | Out-Null
    [IO.File]::WriteAllText(
        (Join-Path $RepoEmbed 'src\lib.rs'),
        @'
pub struct ScoreLimits { pub minimum: i32, pub maximum: i32 }
pub struct ScoreSample { pub label: String, pub value: i32 }
pub struct ScoreStats { pub mean: f64, pub count: usize }
pub struct TrendWindow { pub samples: Vec<i32>, pub capacity: usize }
'@,
        [Text.UTF8Encoding]::new($false)
    )
    [IO.File]::WriteAllText(
        (Join-Path $RepoBrief 'src\lib.rs'),
        @'
pub fn apply_limit(value: i32) -> i32 { value.clamp(0, 100) }
pub fn process_value(value: i32) -> i32 { apply_limit(value) }
pub fn normalize_score(value: i32) -> i32 { value.max(0) }
'@,
        [Text.UTF8Encoding]::new($false)
    )

    $indexEmbed = Start-GreppyProcess @('--root', $RepoEmbed, 'index', $RepoEmbed)
    Complete-GreppyProcess $indexEmbed (Join-Path $Work 'out\index-embed.txt') $false
    if ($indexEmbed.Process.ExitCode -ne 0) { throw 'embedding fixture index failed' }
    $indexBrief = Start-GreppyProcess @('--root', $RepoBrief, 'index', $RepoBrief)
    Complete-GreppyProcess $indexBrief (Join-Path $Work 'out\index-brief.txt') $false
    if ($indexBrief.Process.ExitCode -ne 0) { throw 'brief fixture index failed' }

    $doctor = Invoke-GreppyJson @(
        '--root', $RepoEmbed, 'doctor', '--json'
    ) (Join-Path $Work 'out\doctor.json') @(0, 1)
    $EmbedEndpoint = [string]$doctor.inference.daemons.embedding.endpoint
    $SummaryEndpoint = [string]$doctor.inference.daemons.summary.endpoint
    [void][GreppyPipeClient]::PipeName($EmbedEndpoint)
    [void][GreppyPipeClient]::PipeName($SummaryEndpoint)

    Write-Section 'first graph command prewarm daemon readiness'
    # Indexing owns its model only for the duration of the index operation.
    # The public async daemon-prewarm contract begins when a graph command
    # opens an indexed store that already contains vectors. Drive that exact
    # contract before probing the private diagnostic endpoint; merely waiting
    # after `index` would test a lifecycle the CLI does not promise.
    $script:PrewarmNavigation = Start-GreppyProcess @(
        '--root', $RepoEmbed, 'search-symbol', 'ScoreLimits', '--json'
    )
    Wait-For 'embedding daemon endpoint after first graph-command prewarm' 120 {
        $prewarmStatus = Get-DaemonStatus $EmbedEndpoint
        $prewarmStatus.state -in @('starting', 'loading', 'ready', 'evicted', 'faulted')
    }
    # The prewarm is explicitly asynchronous. A compact graph lookup must not
    # stay alive for the model TTL merely because its detached daemon remains
    # healthy. Bound this separately so failure diagnostics retain both live
    # processes and their exact command lines instead of observing them only
    # after the TTL has expired.
    Wait-For 'first graph command process and redirected streams to close after async daemon prewarm' 30 {
        $script:PrewarmNavigation.Process.HasExited -and
            $script:PrewarmNavigation.Stdout.IsCompleted -and
            $script:PrewarmNavigation.Stderr.IsCompleted
    }
    Complete-GreppyProcess $script:PrewarmNavigation `
        (Join-Path $Work 'out\prewarm-navigation.json') $true
    $prewarmStatus = Get-DaemonStatus $EmbedEndpoint
    Write-Host "embedding-prewarm=$($prewarmStatus | ConvertTo-Json -Depth 5 -Compress)"
    if ($prewarmStatus.state -eq 'faulted') {
        throw "embedding daemon prewarm faulted: $($prewarmStatus.last_error)"
    }

    Write-Section 'spawn, protocol, malformed, oversize and slow-client contract'
    $warmup = Invoke-Semantic 'warmup' (New-Query 'warmup')
    $embedModelKey = Get-EmbeddingModelKey $warmup (Join-Path $Work 'store')
    Wait-For 'embedding daemon to become ready' 30 {
        (Get-DaemonStatus $EmbedEndpoint).state -eq 'ready'
    }
    $EmbedPid = [int](Get-DaemonStatus $EmbedEndpoint).daemon_pid
    if (-not (Test-ProcessAlive $EmbedPid)) { throw 'daemon_pid is not live' }

    $ping = Invoke-PipeJson $EmbedEndpoint @{
        protocol = 2; op = 'ping'; request_id = 'windows-sanity-ping'
    }
    if (-not $ping.ok -or $ping.request_id -ne 'windows-sanity-ping') {
        throw 'ping response did not preserve the request identity'
    }
    $oldProtocol = Invoke-PipeJson $EmbedEndpoint @{ protocol = 1; op = 'ping' }
    if ($oldProtocol.error -ne 'protocol-version mismatch') {
        throw 'stale protocol version was not rejected'
    }
    $malformed = [GreppyPipeClient]::RoundTripText(
        $EmbedEndpoint, 'malformed', 'not json', 10000)
    if ($malformed.Error -or
        ($malformed.Response | ConvertFrom-Json).error -ne 'malformed request') {
        throw 'malformed request was not rejected'
    }
    $oversizePayload = [Text.Encoding]::ASCII.GetBytes('x' * (1048576 + 4096))
    $oversize = [GreppyPipeClient]::RoundTripBytes(
        $EmbedEndpoint, 'oversize', $oversizePayload, 30000)
    if ($oversize.Error -or
        ($oversize.Response | ConvertFrom-Json).error -ne 'request too large or incomplete' -or
        $oversize.ElapsedMilliseconds -ge 4000) {
        throw 'oversize request was not rejected promptly'
    }
    if ([int](Get-DaemonStatus $EmbedEndpoint).daemon_pid -ne $EmbedPid) {
        throw 'oversize request replaced the daemon'
    }

    $slow = [GreppyPipeClient]::BeginRaw(
        $EmbedEndpoint, 'slow', [Text.Encoding]::ASCII.GetBytes('{'), 15000)
    [void](Invoke-Semantic 'slow-bypass' (New-Query 'slow-bypass'))
    $slowResult = $slow.GetAwaiter().GetResult()
    if ($slowResult.Error) { throw "slow client failed: $($slowResult.Error)" }
    $slowResponse = $slowResult.Response | ConvertFrom-Json
    if ($slowResponse.error -ne 'request too large or incomplete' -or
        $slowResult.ElapsedMilliseconds -lt 4000 -or
        $slowResult.ElapsedMilliseconds -ge 15000) {
        throw 'slow client did not receive the bounded read-timeout rejection'
    }

    Write-Section '32+ clients, one owner and classified queue pressure'
    $children = @()
    for ($index = 0; $index -lt 5; $index++) {
        $children += [pscustomobject]@{
            Handle = Start-GreppyProcess @(
                '--root', $RepoEmbed, 'search', '--json',
                (New-Query "parallel-$index")
            )
            Output = Join-Path $Work "out\parallel-$index.json"
        }
    }
    Wait-For 'an active embedding request' 60 {
        -not [String]::IsNullOrEmpty(
            [string](Get-DaemonStatus $EmbedEndpoint).active_request_id)
    }
    $floodStatusBefore = Get-DaemonStatus $EmbedEndpoint
    $flood = Convert-Burst (
        [GreppyPipeClient]::BurstEmbedding(
            $EmbedEndpoint,
            'flood',
            48,
            20000,
            [string]$warmup.prompt_version,
            $embedModelKey)
    )
    $floodStatusAfter = Get-DaemonStatus $EmbedEndpoint
    $floodCapacity = @($flood | Where-Object { $_.capacity -eq $true }).Count
    $floodInference = @($flood | Where-Object { $_.inference_ok -eq $true }).Count
    $floodClientErrors = @($flood | Where-Object { $_.client_error }).Count
    $floodUnexpected = @(
        $flood | Where-Object {
            $_.capacity -ne $true -and $_.inference_ok -ne $true
        }
    ).Count
    $floodRejectedDelta =
        [int]$floodStatusAfter.rejected_requests -
        [int]$floodStatusBefore.rejected_requests
    $floodEvidence = [ordered]@{
        before_rejected_requests = [int]$floodStatusBefore.rejected_requests
        after_rejected_requests = [int]$floodStatusAfter.rejected_requests
        rejected_delta = $floodRejectedDelta
        capacity = $floodCapacity
        inference_ok = $floodInference
        unexpected = $floodUnexpected
        echo_ok = @($flood | Where-Object { $_.echo_ok -eq $true }).Count
        client_errors = $floodClientErrors
        results = @($flood)
    }
    $floodEvidenceJson = $floodEvidence | ConvertTo-Json -Depth 6 -Compress
    [IO.File]::WriteAllText(
        (Join-Path $Work 'out\flood-evidence.json'),
        $floodEvidenceJson,
        [Text.UTF8Encoding]::new($false)
    )
    Write-Host "flood-evidence=$floodEvidenceJson"
    if ($floodCapacity -lt 1) {
        if ($floodRejectedDelta -gt 0) {
            throw "daemon counted capacity rejections but no client received a classified response: $floodEvidenceJson"
        }
        throw "busy 48-client flood produced no classified capacity response: $floodEvidenceJson"
    }
    if ($floodInference -lt 1 -or $floodClientErrors -ne 0 -or $floodUnexpected -ne 0) {
        throw "busy 48-client flood returned incomplete or invalid outcomes: $floodEvidenceJson"
    }
    foreach ($child in $children) {
        Complete-GreppyProcess $child.Handle $child.Output $true
    }
    $burst = Convert-Burst (
        [GreppyPipeClient]::BurstEmbedding(
            $EmbedEndpoint,
            'burst',
            26,
            30000,
            [string]$warmup.prompt_version,
            $embedModelKey)
    )
    $burstUnexpected = @(
        $burst | Where-Object {
            $_.capacity -ne $true -and $_.inference_ok -ne $true
        }
    ).Count
    if (@($burst | Where-Object client_error).Count -ne 0 -or
        $burstUnexpected -ne 0 -or
        @($burst | Where-Object inference_ok).Count -lt 1) {
        throw "26-client burst returned incomplete or invalid outcomes: $($burst | ConvertTo-Json -Depth 6 -Compress)"
    }
    if ([int](Get-DaemonStatus $EmbedEndpoint).daemon_pid -ne $EmbedPid) {
        throw 'daemon owner changed during the concurrent-client gate'
    }

    Write-Section 'forced termination during cold load and automatic respawn'
    Stop-Daemon $EmbedPid
    $killChildren = @()
    for ($index = 0; $index -lt 3; $index++) {
        $killChildren += Start-GreppyProcess @(
            '--root', $RepoEmbed, 'search', '--json', (New-Query "kill-$index")
        )
    }
    Wait-For 'cold replacement daemon with an active request' 60 {
        $status = Get-DaemonStatus $EmbedEndpoint
        [int]$status.daemon_pid -ne $EmbedPid -and
            -not [String]::IsNullOrEmpty([string]$status.active_request_id)
    }
    $ColdPid = [int](Get-DaemonStatus $EmbedEndpoint).daemon_pid
    Stop-Daemon $ColdPid
    foreach ($child in $killChildren) {
        Complete-GreppyProcess $child (Join-Path $Work "out\killed-$($child.Process.Id).json") $false
    }
    [void](Invoke-Semantic 'kill-recovery' (New-Query 'kill-recovery'))
    Wait-For 'healthy daemon after forced termination' 30 {
        $status = Get-DaemonStatus $EmbedEndpoint
        [int]$status.daemon_pid -ne $ColdPid -and $status.state -eq 'ready'
    }
    $EmbedPid = [int](Get-DaemonStatus $EmbedEndpoint).daemon_pid

    Write-Section 'summary daemon, queue pressure and frame limit'
    $summaryContract = Get-SummaryModelContract $doctor (Join-Path $Work 'store')
    $brief = Start-GreppyProcess @(
        '--root', $RepoBrief, 'brief', 'apply_limit', '--json'
    )
    Wait-For 'an active summary request' 120 {
        -not [String]::IsNullOrEmpty(
            [string](Get-DaemonStatus $SummaryEndpoint).active_request_id)
    }
    $summaryFlood = Convert-Burst (
        [GreppyPipeClient]::BurstSummary(
            $SummaryEndpoint,
            'summary-flood',
            48,
            30000,
            $summaryContract.prompt_version,
            $summaryContract.filter_version,
            $summaryContract.model_key)
    )
    $summaryCapacity = @($summaryFlood | Where-Object capacity).Count
    $summaryInference = @($summaryFlood | Where-Object inference_ok).Count
    $summaryUnexpected = @(
        $summaryFlood | Where-Object {
            $_.capacity -ne $true -and $_.inference_ok -ne $true
        }
    ).Count
    if ($summaryCapacity -lt 1 -or $summaryInference -lt 1 -or
        $summaryUnexpected -ne 0 -or
        @($summaryFlood | Where-Object client_error).Count -ne 0) {
        throw "summary flood returned incomplete or invalid outcomes: $($summaryFlood | ConvertTo-Json -Depth 6 -Compress)"
    }
    $briefPath = Join-Path $Work 'out\brief.json'
    Wait-GreppyChild $brief 'summary brief command'
    $briefStdout = $brief.Stdout.GetAwaiter().GetResult()
    $briefStderr = $brief.Stderr.GetAwaiter().GetResult()
    [IO.File]::WriteAllText($briefPath, $briefStdout, [Text.UTF8Encoding]::new($false))
    if ($brief.Process.ExitCode -ne 0) { throw "brief failed: $briefStderr" }
    $briefJson = $briefStdout | ConvertFrom-Json
    if ($briefJson.schema_version -ne 'greppy.brief.v1' -or
        $briefJson.status -ne 'ok' -or
        @($briefJson.definitions).Count -lt 1 -or
        [String]::IsNullOrEmpty([string]$briefJson.definitions[0].summary)) {
        throw 'brief JSON did not satisfy the packaged summary contract'
    }
    $SummaryPid = [int](Get-DaemonStatus $SummaryEndpoint).daemon_pid
    $summaryOversizePayload = [Text.Encoding]::ASCII.GetBytes('x' * (262144 + 4096))
    $summaryOversize = [GreppyPipeClient]::RoundTripBytes(
        $SummaryEndpoint, 'summary-oversize', $summaryOversizePayload, 30000)
    if ($summaryOversize.Error -or
        ($summaryOversize.Response | ConvertFrom-Json).error -ne 'request too large or incomplete') {
        throw 'summary daemon accepted an oversize request'
    }

    Write-Section 'model eviction, reload and idle process exit'
    Stop-Daemon $EmbedPid
    $env:GREPPY_EMBED_DAEMON_MODEL_TTL_S = '2'
    $env:GREPPY_EMBED_DAEMON_EXIT_TTL_S = '8'
    [void](Invoke-Semantic 'evict-warmup' (New-Query 'evict-warmup'))
    $EmbedPid = [int](Get-DaemonStatus $EmbedEndpoint).daemon_pid
    Wait-For 'embedding model eviction' 30 {
        (Get-DaemonStatus $EmbedEndpoint).state -eq 'evicted'
    }
    [void](Invoke-Semantic 'evict-reload' (New-Query 'evict-reload'))
    Wait-For 'short-TTL embedding daemon idle exit' 60 {
        -not (Test-ProcessAlive $EmbedPid)
    }
    Wait-For 'short-TTL summary daemon idle exit' 90 {
        -not (Test-ProcessAlive $SummaryPid)
    }

    Write-Host ""
    Write-Host "Windows release daemon stress passed: $Binary"
}
catch {
    Write-FailureDiagnostics
    throw
}
finally {
    foreach ($childProcess in @($ChildProcesses)) {
        if (-not $childProcess.HasExited) {
            Stop-Process -Id $childProcess.Id -Force -ErrorAction SilentlyContinue
        }
    }
    foreach ($endpointName in @('EmbedEndpoint', 'SummaryEndpoint')) {
        $endpointVariable = Get-Variable -Name $endpointName -ErrorAction SilentlyContinue
        if ($null -ne $endpointVariable) {
            try {
                [void](Get-DaemonStatus ([string]$endpointVariable.Value))
            }
            catch {
                # An absent pipe is already clean; known owners are stopped below.
            }
        }
    }
    foreach ($daemonPid in @($DaemonPids)) {
        Stop-Process -Id $daemonPid -Force -ErrorAction SilentlyContinue
    }
}
