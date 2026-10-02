param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$demoPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $demoPython)) {
    throw 'Missing .venv. Create the environment and install requirements.txt as described in README.'
}
Push-Location -LiteralPath $projectRoot
try {
    & $demoPython -X utf8 -B -m scripts.check_steel_demo_environment
    if ($LASTEXITCODE -ne 0) { throw 'Demo startup checks failed. See the message above.' }
    & $demoPython -X utf8 -B -m scripts.serve_steel_demo --port $Port
} finally {
    Pop-Location
}
