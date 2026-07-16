#!/usr/bin/env pwsh
# Render a trained asset.
#   .\render.ps1 tomo  <asset> [flags...]   # tomography (constant illumination)
#   .\render.ps1 ms    <asset> [flags...]   # multiple scattering (env-map lit)
#   .\render.ps1 video <asset> [flags...]   # turntable / rotating-envmap video
#   .\render.ps1 asset <asset> [flags...]   # raw gfields.render (no mode preset)
# tomo/ms preset flags come from gfields.presets; extra flags override them.
# See `python -m gfields.render --help` and `... video --help`.
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
& $py -m gfields.render @args
exit $LASTEXITCODE
