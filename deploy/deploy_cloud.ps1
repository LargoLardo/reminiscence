$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "deploy_modal.ps1")
& (Join-Path $PSScriptRoot "deploy_vercel.ps1")
