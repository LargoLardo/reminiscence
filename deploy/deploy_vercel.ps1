param(
    [string]$ModalUrl = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendDir = Join-Path $repoRoot "frontend"
$statePath = Join-Path $PSScriptRoot ".modal-state.json"

if (-not $ModalUrl -and (Test-Path -LiteralPath $statePath)) {
    $ModalUrl = (Get-Content -LiteralPath $statePath | ConvertFrom-Json).api_url
}
if (-not $ModalUrl) {
    throw "Pass -ModalUrl or deploy Modal first with deploy/deploy_modal.ps1."
}
$npxCommand = Get-Command npx.cmd -ErrorAction SilentlyContinue
if (-not $npxCommand) {
    $npxFallback = "C:\Program Files\nodejs\npx.cmd"
    if (Test-Path -LiteralPath $npxFallback) {
        $npxCommand = Get-Item -LiteralPath $npxFallback
    }
}
if (-not $npxCommand) {
    throw "npx is not installed. Install Node.js LTS first."
}
$npx = $npxCommand.Source

& $npx --yes vercel link --cwd $frontendDir --yes --project reminiscence
if ($LASTEXITCODE -ne 0) {
    throw "Could not link the Vercel project."
}

& $npx --yes vercel env add VITE_API_BASE_URL production --value $ModalUrl --yes --force --cwd $frontendDir
if ($LASTEXITCODE -ne 0) {
    throw "Could not configure the production Modal API URL in Vercel."
}

& $npx --yes vercel --cwd $frontendDir --prod --yes
if ($LASTEXITCODE -ne 0) {
    throw "Vercel deployment failed."
}
