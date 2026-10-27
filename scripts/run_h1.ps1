param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('pems03', 'metr_la', 'bj500', 'bike_inflow', 'bike_outflow')]
    [string]$Dataset,
    [Parameter(Mandatory = $true)]
    [ValidateSet('baseline', 'improved', 'evaluate')]
    [string]$Stage,
    [Parameter(Mandatory = $true)][string]$Output,
    [ValidateSet(2021, 2022, 2023)][int]$Seed = 2021,
    [ValidateSet('full', 'global_long', 'no_utility')][string]$Arm = 'full',
    [string]$Baseline,
    [string]$NumericCalibration,
    [string]$CheckpointDir,
    [ValidateSet('baseline', 'improved')][string]$EvalModel = 'improved',
    [ValidateRange(1, 2147483647)][int]$MaxWindows,
    [int[]]$Origins,
    [switch]$SavePredictions,
    [switch]$ReplayTrainingRng,
    [string]$RngCheckpoint,
    [switch]$Resume,
    [string]$DataRoot = 'dataset',
    [string]$BertPath = 'huggingface/google_bert',
    [string]$PythonPath = 'python',
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'h1_common.ps1')

if ($Stage -eq 'improved' -and -not $Baseline) { throw 'Improved training requires -Baseline.' }
if ($Stage -eq 'evaluate' -and -not $CheckpointDir) { throw 'Evaluation requires -CheckpointDir.' }
if ($Arm -ne 'full' -and ($Dataset -ne 'bike_inflow' -or $Stage -eq 'baseline')) {
    throw 'Ablation arms are only supported for BIKE-Inflow improved/evaluate.'
}

$cli = @('research/run_h1.py', '--dataset', $Dataset, '--stage', $Stage,
    '--seed', "$Seed", '--arm', $Arm, '--output', (Resolve-H1Path $Output),
    '--data-root', (Resolve-H1Path $DataRoot), '--bert-path', (Resolve-H1Path $BertPath))
foreach ($item in @(
    @('--baseline', $Baseline), @('--numeric-calibration', $NumericCalibration),
    @('--checkpoint-dir', $CheckpointDir), @('--rng-checkpoint', $RngCheckpoint)
)) {
    if ($item[1]) { $cli += @($item[0], (Resolve-H1Path $item[1])) }
}
if ($Stage -eq 'evaluate') { $cli += @('--eval-model', $EvalModel) }
if ($PSBoundParameters.ContainsKey('MaxWindows')) { $cli += @('--max-windows', "$MaxWindows") }
if ($Origins) {
    $cli += '--origins'
    $cli += @($Origins | ForEach-Object { "$_" })
}
if ($SavePredictions) { $cli += '--save-predictions' }
if ($ReplayTrainingRng) { $cli += '--replay-training-rng' }
if ($Resume) { $cli += '--resume' }

# Python owns protocol, checkpoint and overwrite validation. No budget overrides.
Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments $cli
