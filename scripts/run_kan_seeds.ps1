param(
    [int[]]$Seeds = @(2021,42,7),
    [ValidateSet('both','kan','mlp')][string]$Heads = 'both',
    [string]$OutputRoot = 'outputs/kan_queue',
    [string]$PythonPath = 'python',
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'h1_common.ps1')
$cli = @('scripts/training_runs.py','kan','--output-root',(Resolve-H1Path $OutputRoot),'--heads',$Heads,'--seeds')
$cli += @($Seeds | ForEach-Object { "$_" })
if ($PlanOnly) { $cli += '--plan-only' }
Invoke-H1Python -PythonPath $PythonPath -Arguments $cli
