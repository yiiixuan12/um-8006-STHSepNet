param(
    [ValidateSet('pems03','metr_la','bj500','bike_inflow','bike_outflow')][string[]]$Datasets = @('bike_inflow','bike_outflow','pems03','bj500','metr_la'),
    [int[]]$Seeds = @(2021,2022,2023),
    [string]$OutputRoot = 'results_coursework/run_jinqi_baseline',
    [string]$PythonPath = 'python',
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'h1_common.ps1')
$cli = @('scripts/training_runs.py','baseline','--output-root',(Resolve-H1Path $OutputRoot),'--datasets') + $Datasets
$cli += '--seeds'
$cli += @($Seeds | ForEach-Object { "$_" })
if ($PlanOnly) { $cli += '--plan-only' }
Invoke-H1Python -PythonPath $PythonPath -Arguments $cli
