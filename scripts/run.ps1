Param(
	[Parameter(ValueFromRemainingArguments=$true)]
	[String[]]$args
)

# Run the application using the existing .venv without rebuilding.
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Push-Location $RepoRoot

$VenvActivate = Join-Path $RepoRoot '.venv\Scripts\Activate.ps1'
if (-not (Test-Path $VenvActivate)) {
	Write-Error "Virtual environment not found at $VenvActivate. Run scripts\build_and_run.ps1 first."
	Pop-Location
	exit 1
}

Write-Host "Activating venv at $VenvActivate..."
# Dot-source the activate script so the current session uses the venv
. $VenvActivate

Write-Host 'Launching application (no rebuild)...'
python -m app.main @args
$rc = $LASTEXITCODE

Pop-Location
exit $rc
