param(
    [switch]$Update
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $ProjectRoot "environment.yml"

if ($Update) {
    conda env update -f $EnvFile --prune
} else {
    conda env create -f $EnvFile
}

conda run -n omni-vsr pip install -e $ProjectRoot
Write-Host "Conda environment ready: omni-vsr"
