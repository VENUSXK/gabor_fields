# Reproducible environment for volumetric_primitives via uv:
#   1. create a uv-managed .venv (Python 3.13),
#   2. install the gfields package (editable),
#   3. compile the vendored Mitsuba 3 fork against that venv and wire it in,
#   4. validate.
#
#   .\install.ps1                 # default: scalar_rgb + cuda_ad_rgb variants
#   .\install.ps1 -Extras         # + torch / lpips / wandb / scikit-image / flip
#   .\install.ps1 --variants scalar_rgb,cuda_ad_rgb,llvm_ad_rgb   # add CPU fallback
# Extra flags are forwarded to gfields/build_mitsuba.py (--jobs, ...).
[CmdletBinding()]
param(
    [switch]$Extras,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Passthru
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$PyVer = if ($env:VOLPRIM_PYTHON_VERSION) { $env:VOLPRIM_PYTHON_VERSION } else { "3.13" }
$Venv  = Join-Path $PSScriptRoot ".venv"

function Need($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) { throw "ERROR: '$name' not found. $hint" }
}

# Ensure uv (the Python package/venv manager this script drives) is available.
# The official installer drops the binary in %USERPROFILE%\.local\bin, which is
# not on PATH in a fresh shell, so first try to pick up an existing install from
# there before downloading — avoids clobbering (and locking) a working uv.exe.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $UvBin = Join-Path $env:USERPROFILE ".local\bin"
    if (-not (Test-Path (Join-Path $UvBin "uv.exe"))) {
        Write-Host "'uv' not found; installing via https://astral.sh/uv/install.ps1 ..."
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
    if (Test-Path (Join-Path $UvBin "uv.exe")) {
        # Add to this session's PATH, and persist to the User PATH so future
        # shells (and other tools) find uv without re-running this. The persist
        # is idempotent — skip if .local\bin is already on the stored User PATH.
        $env:Path = "$UvBin;$env:Path"
        $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if (($UserPath -split ';') -notcontains $UvBin) {
            Write-Host "Adding $UvBin to your User PATH (permanent)."
            $NewPath = if ($UserPath) { "$UvBin;$UserPath" } else { $UvBin }
            [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
        }
    }
    Need uv "Install it manually: https://docs.astral.sh/uv/getting-started/installation/  (irm https://astral.sh/uv/install.ps1 | iex), then reopen your shell."
}
Need cmake "Required to compile the Mitsuba fork."
Need git   ""
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Warning "'nvidia-smi' not found; the cuda_ad_rgb variant builds but needs an NVIDIA driver at runtime."
}

# Ensure the pinned submodules are present. Mitsuba (+ its ext/drjit at the
# pinned commit, hence --recursive) supplies the renderer and Dr.Jit; volprim
# supplies the `volprim.*` runtime package that gfields imports.
if (-not (Test-Path "submodules\mitsuba3\CMakeLists.txt")) {
    Write-Host "Fetching Mitsuba submodule (+ ext/drjit) ..."
    git submodule update --init --recursive submodules/mitsuba3
}
if (-not (Test-Path "submodules\volprim\setup.py")) {
    Write-Host "Fetching volprim submodule ..."
    git submodule update --init submodules/volprim
}

Write-Host "== 1/4  create uv venv (python $PyVer) -> .venv =="
uv venv --python $PyVer $Venv
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = Join-Path $Venv "bin/python" }

Write-Host "== 2/4  install python packages (editable) =="
uv pip install --python $Py -e .
# volprim provides the `volprim.*` package gfields imports. Install it editable
# with --no-deps so its 'mitsuba>=3.9' / 'drjit>=1.4' pins do NOT pull the stock
# PyPI wheels over the locally compiled fork wired in step 3 (its other deps —
# numpy/matplotlib/plyfile/ipywidgets/rich — are already covered by gfields).
uv pip install --python $Py --no-deps -e submodules/volprim
if ($Extras) {
    Write-Host "   (+ extras: torch / lpips / wandb / metrics)"
    uv pip install --python $Py -e ".[metrics,training,test]"
}

Write-Host "== 3/4  compile the Mitsuba fork and wire it into the venv =="
& $Py gfields/build_mitsuba.py --python $Py @Passthru
if ($LASTEXITCODE -ne 0) { throw "Mitsuba build failed (exit $LASTEXITCODE)." }

Write-Host "== 4/4  validate =="
& $Py -c "import mitsuba as mi; mi.set_variant('cuda_ad_rgb'); import gfields, gfields.cli; print('OK: mitsuba', mi.__version__, '| gfields import OK')"

Write-Host ""
Write-Host "Environment ready in .venv ."
Write-Host "  Activate:  .venv\Scripts\Activate.ps1"
Write-Host "  Train:     examples\train.ps1 wdas8   (launchers auto-detect .venv)"
Write-Host "  Override interpreter anywhere with `$env:VOLPRIM_PYTHON."
