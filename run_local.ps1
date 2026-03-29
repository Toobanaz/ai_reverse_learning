$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendCommand = "Set-Location '$projectRoot'; python app.py"
$frontendCommand = "Set-Location '$projectRoot'; npm run dev"

Write-Host "Starting backend in a new PowerShell window..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $backendCommand
)

Start-Sleep -Seconds 1

Write-Host "Starting frontend in a new PowerShell window..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $frontendCommand
)

Write-Host "Launched backend and frontend windows." -ForegroundColor Green
