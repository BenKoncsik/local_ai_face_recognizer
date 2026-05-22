#Requires -Version 5.1
# Build and run script for Windows (PowerShell)

$ErrorActionPreference = 'Stop'

$RepoRoot  = (Resolve-Path "$PSScriptRoot\..").Path
$VenvDir   = Join-Path $RepoRoot '.venv'
$ModelsDir = Join-Path $RepoRoot 'models'
$ConfigFile = Join-Path $RepoRoot 'config.yaml'

Push-Location $RepoRoot

# -- Find Python 3.11+ --------------------------------------------------------
Write-Host '==> Checking Python...'

$Python = $null
foreach ($cmd in @('python3.13','python3.12','python3.12.6','python3.11','python3','python')) {
    if ($Python) { break }
    $exe = Get-Command $cmd -ErrorAction SilentlyContinue
    if (-not $exe) { continue }
    try {
        $ok = & $exe.Source -c "import sys; print('ok' if sys.version_info >= (3,11) else 'old')" 2>$null
    } catch {
        # Failed to run this candidate (e.g. App Execution Alias or launcher); skip it
        continue
    }
    if ($ok -and $ok.ToString().Trim() -eq 'ok') { $Python = $exe.Source }
}

if (-not $Python) {
    Write-Error 'ERROR: Python 3.11+ not found. Install from https://www.python.org/downloads/ and add to PATH.'
    exit 1
}

$PyVer = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "    Using Python $PyVer ($Python)"

# -- Virtual environment ------------------------------------------------------
Write-Host "==> Setting up virtual environment at $VenvDir ..."

$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
if (Test-Path $VenvPython) {
    $VenvVer = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($VenvVer -ne $PyVer) {
        Write-Host "    Stale venv (Python $VenvVer) - rebuilding with $PyVer..."
        Remove-Item $VenvDir -Recurse -Force
    }
}

if (-not (Test-Path $VenvPython)) {
    & $Python -m venv $VenvDir
}

& "$VenvDir\Scripts\Activate.ps1"

# -- Dependencies -------------------------------------------------------------
Write-Host '==> Installing / updating dependencies...'
python -m pip install --upgrade pip setuptools wheel --quiet
python -m pip install -e ".[dev]" --quiet

Write-Host '==> Trying optional TPU packages (ai-edge-litert)...'
try {
    # Run pip install for optional tflite extras. Some Python versions (or pip/resolution)
    # may return a non-zero exit code; don't let that stop the whole script.
    & python -m pip install -e ".[tflite]" --quiet 2>$null
} catch {
    Write-Host "    WARNING: TPU support unavailable for Python $PyVer."
}

# -- Model downloads ----------------------------------------------------------
Write-Host '==> Checking model files...'
if (-not (Test-Path $ModelsDir)) { New-Item $ModelsDir -ItemType Directory | Out-Null }

function Download-IfMissing {
    param([string]$Dest, [string]$Url, [string]$Label)
    if (Test-Path $Dest) { Write-Host "    [ok] $Label"; return }
    Write-Host "    Downloading $Label ..."
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
        Write-Host "    [ok] $Label downloaded"
    } catch {
        Write-Host "    WARNING: Failed to download $Label"
    }
}

Download-IfMissing `
    (Join-Path $ModelsDir 'deploy.prototxt') `
    'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt' `
    'deploy.prototxt (CPU detector)'

Download-IfMissing `
    (Join-Path $ModelsDir 'res10_300x300_ssd_iter_140000.caffemodel') `
    'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel' `
    'res10_300x300_ssd_iter_140000.caffemodel (CPU detector)'

Download-IfMissing `
    (Join-Path $ModelsDir 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite') `
    'https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite' `
    'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite (Coral detector)'

# MobileFaceNet - try mirrors
$FaceNet = Join-Path $ModelsDir 'mobilefacenet.tflite'
if (-not (Test-Path $FaceNet)) {
    Write-Host '    Downloading MobileFaceNet embedding model (trying mirrors)...'
    $MfnUrls = @(
        'https://github.com/shubham0204/FaceRecognition_With_FaceNet_Android/raw/master/app/src/main/assets/mobile_face_net.tflite',
        'https://github.com/estebanuri/face_recognition/raw/master/android/face-recognition/app/src/main/assets/mobile_face_net.tflite',
        'https://github.com/Martlgap/FaceIDLight/raw/main/facelib/models/pretrained/mobileFaceNet.tflite'
    )
    foreach ($url in $MfnUrls) {
        Write-Host "    -> $url"
        try {
            Invoke-WebRequest -Uri $url -OutFile "$FaceNet.tmp" -UseBasicParsing
            Move-Item "$FaceNet.tmp" $FaceNet -Force
            Write-Host '    [ok] mobilefacenet.tflite downloaded'
            break
        } catch {
            Remove-Item "$FaceNet.tmp" -ErrorAction SilentlyContinue
        }
    }
    if (-not (Test-Path $FaceNet)) {
        Write-Host '    WARNING: Could not download MobileFaceNet - SFace (OpenCV) fallback will be used.'
    }
}

Download-IfMissing `
    (Join-Path $ModelsDir 'sface.onnx') `
    'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx' `
    'sface.onnx (face recognition, 37 MB)'

# -- Config -------------------------------------------------------------------
if (-not (Test-Path $ConfigFile)) {
    Write-Host '==> Generating config.yaml ...'
    @'
# Auto-generated by build_and_run.ps1 - edit as needed.

detection:
  confidence_threshold: 0.65
  min_face_size: 50
  coral_model_path: models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
  cpu_model_path: models/res10_300x300_ssd_iter_140000.caffemodel

embedding:
  model_path: models/mobilefacenet.tflite
  input_size: [112, 112]
  embedding_dim: 192

clustering:
  epsilon: 0.4
  min_samples: 2
  metric: cosine

storage:
  db_path: data/faces.db
  crops_dir: data/crops

scan:
  image_extensions: [.jpg, .jpeg, .png, .webp]
  worker_threads: 2
  thumbnail_size: [128, 128]
'@ | Set-Content $ConfigFile -Encoding UTF8
    Write-Host '    [ok] config.yaml created'
} else {
    Write-Host '==> config.yaml already exists - skipping generation'
}

# -- Launch -------------------------------------------------------------------
Write-Host '==> Build complete. Launching application...'
python -m app.main @args

Pop-Location
