param(
    [string]$EnvName = "torch_env",
    [string]$CondaRoot = "$env:USERPROFILE\anaconda3"
)

$ErrorActionPreference = "Stop"

$condaExe = Join-Path $CondaRoot "Scripts\conda.exe"
if (-not (Test-Path $condaExe)) {
    throw "conda.exe not found at: $condaExe"
}

Write-Host "[1/6] Initializing conda for PowerShell..." -ForegroundColor Cyan
& $condaExe init powershell | Out-Host

Write-Host "[2/6] Loading conda hook in current session..." -ForegroundColor Cyan
$condaHook = Join-Path $CondaRoot "shell\condabin\conda-hook.ps1"
& $condaHook

if (-not (conda env list | Select-String -Pattern ("^" + [regex]::Escape($EnvName) + "\\s"))) {
    Write-Host "Conda env '$EnvName' not found. Creating with Python 3.10..." -ForegroundColor Yellow
    conda create -n $EnvName python=3.10 -y
}

Write-Host "[3/6] Activating env: $EnvName" -ForegroundColor Cyan
conda activate $EnvName

Write-Host "[4/6] Installing project dependencies..." -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

$gpuName = $null
try {
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
} catch {
    $gpuName = $null
}

Write-Host "[5/6] Installing PyTorch..." -ForegroundColor Cyan
if ($gpuName) {
    Write-Host "Detected NVIDIA GPU: $gpuName" -ForegroundColor Green
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "No NVIDIA GPU detected by nvidia-smi. Installing CPU build." -ForegroundColor Yellow
    python -m pip install torch torchvision torchaudio
}

Write-Host "[6/6] Validating runtime and system info..." -ForegroundColor Cyan
python -c "import platform, torch; print('Python:', platform.python_version()); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host "`nSystem summary:" -ForegroundColor Cyan
Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory | Format-Table -AutoSize
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | Format-Table -AutoSize
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM | Format-Table -AutoSize

Write-Host "`nSetup finished. Reopen PowerShell if conda command is still not recognized globally." -ForegroundColor Green
