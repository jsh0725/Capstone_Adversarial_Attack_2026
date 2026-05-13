param(
    [string]$EnvName = "torch_env",
    [string]$CondaRoot = "$env:USERPROFILE\anaconda3"
)

$ErrorActionPreference = "Stop"
$condaExe = Join-Path $CondaRoot "Scripts\conda.exe"
if (-not (Test-Path $condaExe)) { throw "conda.exe not found: $condaExe" }

# Load conda in current PowerShell session
$condaHook = Join-Path $CondaRoot "shell\condabin\conda-hook.ps1"
& $condaHook
conda activate $EnvName

$pkgs = @(
    "numpy",
    "scipy",
    "sympy",
    "pandas",
    "scikit-learn"
)

Write-Host "Checking linear-algebra related packages in '$EnvName'..." -ForegroundColor Cyan
$missing = @()
foreach ($p in $pkgs) {
    python -c "import importlib.util as u; import sys; sys.exit(0 if u.find_spec('$p') else 1)"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] $p" -ForegroundColor Green
    } else {
        Write-Host "[MISS] $p" -ForegroundColor Yellow
        $missing += $p
    }
}

if ($missing.Count -gt 0) {
    Write-Host "`nInstalling missing packages: $($missing -join ', ')" -ForegroundColor Cyan
    python -m pip install --upgrade pip
    python -m pip install $missing
} else {
    Write-Host "`nAll target packages are already installed." -ForegroundColor Green
}

Write-Host "`nFinal version check:" -ForegroundColor Cyan
python -c "import numpy, scipy, sympy, pandas, sklearn; print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('sympy', sympy.__version__); print('pandas', pandas.__version__); print('scikit-learn', sklearn.__version__)"
