$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$repoRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $PSScriptRoot ".modal-state.json"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not installed or is not on PATH."
}
try {
    & python -c "import modal" 2>$null
}
catch {
    throw "Modal CLI is not installed. Run: python -m pip install -r deploy\requirements.txt"
}
if ($LASTEXITCODE -ne 0) {
    throw "Modal CLI is not installed. Run: python -m pip install -r deploy\requirements.txt"
}

Push-Location $repoRoot
try {
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $deployOutput = & python -m modal deploy modal_app.py 2>&1
    $deployExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    $deployOutput | Write-Host
    if ($deployExitCode -ne 0) {
        throw "Modal deployment failed."
    }

    $endpointMatches = [regex]::Matches(
        ($deployOutput -join "`n"),
        "https://[a-zA-Z0-9-]+\.modal\.run"
    )
    if ($endpointMatches.Count -eq 0) {
        throw "Modal deployed, but its web endpoint could not be read from the CLI output."
    }

    $endpoint = $endpointMatches[$endpointMatches.Count - 1].Value
    @{
        api_url = $endpoint
        deployed_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $statePath

    @"

Modal is deployed.
API URL: $endpoint

The API URL is saved locally in deploy/.modal-state.json.
"@ | Write-Host
}
finally {
    Pop-Location
}
