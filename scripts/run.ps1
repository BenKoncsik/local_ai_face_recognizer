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

# Download mobilefacenet.tflite if missing (build_and_run.ps1 normally handles this,
# but run.ps1 is used after an initial setup to skip the full rebuild step).
$FaceNet = Join-Path $RepoRoot 'models\mobilefacenet.tflite'
if (-not (Test-Path $FaceNet)) {
    Write-Host 'mobilefacenet.tflite missing — downloading embedding model...'
    New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot 'models') | Out-Null
    $MfnUrls = @(
        'https://github.com/MCarlomagno/FaceRecognitionAuth/raw/refs/heads/master/assets/mobilefacenet.tflite',
        'https://github.com/pb-julian/liteface/raw/main/tflite_models/mobilefacenet.tflite',
        'https://github.com/shubham0204/FaceRecognition_With_FaceNet_Android/raw/master/app/src/main/assets/mobile_face_net.tflite'
    )
    $downloaded = $false
    foreach ($url in $MfnUrls) {
        try {
            Invoke-WebRequest -Uri $url -OutFile "$FaceNet.tmp" -UseBasicParsing -ErrorAction Stop
            Move-Item "$FaceNet.tmp" $FaceNet -Force
            Write-Host "[ok] mobilefacenet.tflite downloaded"
            $downloaded = $true
            break
        } catch {
            Remove-Item "$FaceNet.tmp" -ErrorAction SilentlyContinue
        }
    }
    if (-not $downloaded) {
        Write-Warning "Could not download mobilefacenet.tflite — face recognition may not work. Use Settings > AI csomagok to retry."
    }
}

Write-Host 'Launching application (no rebuild)...'
python -m app.main @args
$rc = $LASTEXITCODE

Pop-Location
exit $rc
