param(
    [string]$RunPodApiKey = $env:RUNPOD_API_KEY
)

$ErrorActionPreference = "Stop"
if (-not $RunPodApiKey) {
    throw "Set RUNPOD_API_KEY or pass -RunPodApiKey."
}

$statePath = Join-Path $PSScriptRoot ".runpod-state.json"
if (-not (Test-Path -LiteralPath $statePath)) {
    throw "Missing deploy/.runpod-state.json. Nothing to remove."
}

$state = Get-Content -LiteralPath $statePath | ConvertFrom-Json
Invoke-RestMethod `
    -Method Delete `
    -Uri "https://rest.runpod.io/v1/pods/$($state.podId)" `
    -Headers @{ Authorization = "Bearer $RunPodApiKey" }

Remove-Item -LiteralPath $statePath
Write-Host "Removed RunPod Pod $($state.podId)."
