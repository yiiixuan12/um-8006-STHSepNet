param(
    [string]$PythonPath = 'python',
    [string]$ResultsRoot = 'experiments/h1',
    [switch]$Redraw,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'h1_common.ps1')

$evidence = Resolve-H1Path $ResultsRoot
Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments @(
    'research/check_h1_results.py', '--root', $evidence
)
if ($Redraw) {
    Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments @(
        'research/plot_bike_ablation.py', '--summary', (Join-Path $evidence 'ablation/summary.json'),
        '--output-dir', (Resolve-H1Path 'outputs/h1_ablation')
    )
    Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments @(
        'research/plot_h1_cases.py', '--predictions', (Join-Path $evidence 'cases/paired_24h_predictions.npz'),
        '--report', (Join-Path $evidence 'cases/report.json'),
        '--output', (Resolve-H1Path 'outputs/h1_cases')
    )
}
