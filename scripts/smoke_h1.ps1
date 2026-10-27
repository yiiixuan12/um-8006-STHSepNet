param(
    [string]$PythonPath = 'python',
    [switch]$CheckData,
    [ValidateSet('pems03', 'metr_la', 'bj500', 'bike_inflow', 'bike_outflow')]
    [string]$Dataset = 'pems03',
    [string]$DataRoot = 'dataset',
    [string]$BertPath = 'huggingface/google_bert',
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'h1_common.ps1')

# Default: stdlib-only interface checks, NOT a model/CUDA/accuracy smoke run.
Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments @('research/run_h1.py', '--help')
Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments @(
    '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_h1_release.py', '-v'
)
if ($CheckData) {
    # CPU data check needs full installed dependencies and downloaded CSV/adjacency.
    # The baseline stage avoids requiring a completed frozen baseline to check data.
    Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments @(
        'research/run_h1.py', '--dataset', $Dataset, '--stage', 'baseline',
        '--output', (Resolve-H1Path 'runs/h1/data_check'), '--check-data',
        '--data-root', (Resolve-H1Path $DataRoot), '--bert-path', (Resolve-H1Path $BertPath)
    )
}
