#!/usr/bin/env pwsh
# Train volumetric primitives
#   .\train.ps1 <preset|volume>            # Gaussian base + Gabor residual
#   .\train.ps1 <preset|volume> --gaussian # Gaussian-only (essentially improved Don't splat your Gaussians)
# <preset> is a paper asset name (wdas8, embergen, ...); <volume> is a .npy/.vol/.vdb
# file (give an unknown volume a density scale, e.g. --opacity_scale 12). Every
# extra flag overrides this; see `python -m gfields.train --help`.
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Find-GfieldsPython {
    if ($env:VOLPRIM_PYTHON -and (Test-Path $env:VOLPRIM_PYTHON)) { return $env:VOLPRIM_PYTHON }
    $candidates = @(
        (Join-Path $Repo ".venv\Scripts\python.exe"),
        (Join-Path $Repo ".venv/bin/python")
    )
    if ($env:CONDA_PREFIX) { $candidates += (Join-Path $env:CONDA_PREFIX "python.exe") }
    $candidates += @(
        (Join-Path $env:USERPROFILE "anaconda3\envs\volprim\python.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\envs\volprim\python.exe")
    )
    foreach ($c in $candidates) { if ($c -and (Test-Path $c)) { return $c } }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "Python not found. Run .\install.ps1 or set `$env:VOLPRIM_PYTHON."
}
$py = Find-GfieldsPython

if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = "utf-8" }
if (-not $env:WANDB_MODE) { $env:WANDB_MODE = "disabled" }   # opt in with WANDB_MODE=online
Set-Location $Repo
& $py -m gfields.train @args
exit $LASTEXITCODE
