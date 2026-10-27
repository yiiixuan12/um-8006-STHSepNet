param(
    [ValidateSet('pems03','metr_la','bj500','bike_inflow','bike_outflow')][string[]]$Datasets = @('bike_outflow'),
    [string]$OutputRoot,
    [string]$PythonPath = 'python',
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'h1_common.ps1')
$cli = @('scripts/training_runs.py', 'smoke', '--datasets') + $Datasets
if ($OutputRoot) { $cli += @('--output-root', (Resolve-H1Path $OutputRoot)) }
if ($PlanOnly) { $cli += '--plan-only' }
# The Python queue prints each planned training command without running it.
Invoke-H1Python -PythonPath $PythonPath -Arguments $cli
