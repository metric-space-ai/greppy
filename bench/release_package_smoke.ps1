param(
    [Parameter(Mandatory = $true)][string]$Binary,
    [Parameter(Mandatory = $true)][string]$Work
)

$ErrorActionPreference = 'Stop'
$BinaryDir = Split-Path -Parent ([IO.Path]::GetFullPath($Binary))
$WebRuntimeDist = Join-Path $BinaryDir 'web-runtime'
if (Test-Path -LiteralPath $WebRuntimeDist) {
    throw 'Windows 0.4.0 package must not contain the web-runtime distributable'
}
New-Item -ItemType Directory -Force "$Work/repo/src", "$Work/repo/.git", "$Work/store" | Out-Null
New-Item -ItemType Directory -Force `
    "$Work/store/embedded-model", `
    "$Work/store/models/v1/unmanaged-old", `
    "$Work/store/workspaces/v2/unmanaged-old" | Out-Null
'legacy-0.3.1-cache' | Set-Content -Encoding utf8NoBOM "$Work/store/embedded-model/legacy.marker"
'stale-model-placeholder' | Set-Content -Encoding utf8NoBOM "$Work/store/models/v1/unmanaged-old/model.gguf"
'stale-workspace-placeholder' | Set-Content -Encoding utf8NoBOM "$Work/store/workspaces/v2/unmanaged-old/graph.db"
@'
pub fn apply_limit(value: i32) -> i32 { value.clamp(0, 100) }
pub fn process_value(value: i32) -> i32 { apply_limit(value) }
pub fn normalize_score(value: i32) -> i32 { value.max(0) }
pub fn validate_score(value: i32) -> bool { value <= 100 }
pub fn default_score() -> i32 { 50 }
pub fn minimum_score() -> i32 { 0 }
pub fn maximum_score() -> i32 { 100 }
'@ | Set-Content -Encoding utf8NoBOM "$Work/repo/src/lib.rs"

$env:GREPPY_STORE_DIR = "$Work/store"
$env:GREPPY_EMBED_DAEMON_MODEL_TTL_S = '5'
$env:GREPPY_EMBED_DAEMON_EXIT_TTL_S = '15'
$env:GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S = '5'
$env:GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S = '15'

& $Binary --help | Out-Null
$webUnavailable = & $Binary web status --json
$webExit = $LASTEXITCODE
$webText = $webUnavailable -join "`n"
if ($webExit -ne 31 -or
    -not $webText.Contains('web tool is not available on Windows in 0.4.0')) {
    throw "Windows web scope diagnostic failed: exit=$webExit output=$webText"
}
$doctorRaw = & $Binary --device cpu --root "$Work/repo" doctor --json
$doctorExit = $LASTEXITCODE
if ($doctorExit -ne 0 -and $doctorExit -ne 1) { throw "doctor failed: $doctorExit" }
$doctor = $doctorRaw | ConvertFrom-Json
if ($doctor.command -ne 'doctor' -or $doctor.inference.registry.selected_backend -ne 'cpu') {
    throw 'doctor CPU contract failed'
}

& $Binary --device cpu --root "$Work/repo" index "$Work/repo" | Out-File "$Work/index.txt"
if ($LASTEXITCODE -ne 0) { throw "index failed: $LASTEXITCODE" }
$whereAmI = & $Binary --device cpu --root "$Work/repo" where-am-i
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(($whereAmI -join "`n"))) {
    throw 'packaged legacy-cache index: where-am-i failed'
}
$brief = (& $Binary --device cpu --root "$Work/repo" brief apply_limit --json) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $brief.schema_version -ne 'greppy.brief.v1' -or
    $brief.status -ne 'ok' -or $brief.definitions.Count -lt 1 -or
    [string]::IsNullOrWhiteSpace($brief.definitions[0].signature) -or
    $brief.definitions[0].end_line -lt $brief.definitions[0].start_line -or
    $brief.definitions[0].summary.Count -lt 1 -or
    [string]::IsNullOrWhiteSpace($brief.expand_id)) {
    throw 'brief JSON contract failed'
}
$briefExpanded = (& $Binary --root "$Work/repo" expand $brief.expand_id --json) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $briefExpanded.id -ne $brief.expand_id -or
    -not $briefExpanded.payload_text.Contains('apply_limit')) {
    throw 'brief expand contract failed'
}

$semantic = (& $Binary --device cpu --root "$Work/repo" search --json 'restrict a numeric value to an allowed range') | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $semantic.schema_version -ne 'greppy.semantic-search.v1' -or
    $semantic.status -ne 'ok' -or $semantic.hits.Count -lt 1 -or
    [string]::IsNullOrWhiteSpace($semantic.expand_id)) {
    throw 'semantic-search JSON contract failed'
}
$summaries = @($semantic.hits | Where-Object { $_.summary.Count -gt 0 })
$invalidSpans = @($semantic.hits | Where-Object {
    $_.end_line -lt $_.start_line -or [string]::IsNullOrWhiteSpace($_.signature)
})
if ($summaries.Count -lt 1 -or $invalidSpans.Count -gt 0) {
    throw 'semantic-search signature/summary contract failed'
}
$semanticExpanded = (& $Binary --root "$Work/repo" expand $semantic.expand_id --json) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $semanticExpanded.id -ne $semantic.expand_id -or
    [string]::IsNullOrWhiteSpace($semanticExpanded.payload_text) -or
    $semanticExpanded.payload_json.further_hits -ne $semantic.omitted -or
    $semanticExpanded.payload_json.hits.Count -ne $semantic.omitted) {
    throw 'semantic expand contract failed'
}

Write-Host "release package inference smoke passed: $Binary"
