param(
    [switch]$SkipNpmInstall,
    [switch]$SkipPythonInstall
)

$ErrorActionPreference = "Stop"

function Write-Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Assert-Command($name, $helpText) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "$name is not installed or not on PATH. $helpText"
    }
}

Write-Step "Checking prerequisites"
Assert-Command "python" "Install Python 3.10+ and reopen PowerShell."
Assert-Command "npm" "Install Node.js 18+ and reopen PowerShell."

if (-not (Test-Path ".env")) {
    Write-Step "Creating .env from .env.example"
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env. Add your OPENAI_API_KEY before using AI features." -ForegroundColor Yellow
}
else {
    Write-Step ".env already exists"
}

if (-not $SkipPythonInstall) {
    Write-Step "Installing Python dependencies"
    python -m pip install -r requirements.txt
}

if (-not $SkipNpmInstall) {
    Write-Step "Installing frontend dependencies"
    npm install
}

Write-Step "Setup complete"
Write-Host "Run the backend in one terminal: python app.py" -ForegroundColor Green
Write-Host "Run the frontend in another terminal: npm run dev" -ForegroundColor Green
Write-Host "Or launch both with: .\\run_local.ps1" -ForegroundColor Green
Write-Host "If transcription fails, install ffmpeg from https://www.ffmpeg.org/download.html and make sure it is on PATH." -ForegroundColor Yellow
