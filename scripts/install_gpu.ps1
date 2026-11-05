<#
.SYNOPSIS
    Switch a venv to the NVIDIA GPU build of local embeddings, on Windows.

.DESCRIPTION
    pip can't do this in one step. chromadb requires the CPU onnxruntime, which shares a folder with
    onnxruntime-gpu and overwrites it, so embeddings would silently stay on the CPU. Stop the server
    first, because Windows won't let pip replace a DLL that's in use.

.PARAMETER Python
    The venv's python.exe. Defaults to .venv in the project root.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_gpu.ps1
#>
param(
    [string]$Python = (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe")
)

function Invoke-Pip {
    param([string[]]$Arguments)
    Write-Host "pip $($Arguments -join ' ')"
    & $Python -m pip @Arguments --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) {
        Write-Host "pip failed. If a file was in use, stop the server and run this script again." -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path $Python)) {
    Write-Host "Python not found: $Python" -ForegroundColor Red
    exit 1
}

Set-Location (Join-Path $PSScriptRoot "..")
Invoke-Pip @("uninstall", "-y", "fastembed", "onnxruntime")
Invoke-Pip @("install", "-r", "requirements-gpu.txt")
Invoke-Pip @("uninstall", "-y", "onnxruntime", "onnxruntime-gpu")
Invoke-Pip @("install", "--no-deps", "onnxruntime-gpu")

# Check the result, because the failure this script prevents is otherwise silent.
& $Python -c "import onnxruntime as o; o.preload_dlls(); assert 'CUDAExecutionProvider' in o.get_available_providers(), 'CUDA missing'"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installed, but CUDA isn't available. Check the NVIDIA driver with nvidia-smi." -ForegroundColor Red
    exit 1
}
Write-Host "Done: onnxruntime can use CUDA. pip check will say chromadb requires onnxruntime; that's expected." -ForegroundColor Green
