param(
    [string]$PythonPath = 'python'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonExe = (Get-Command $PythonPath -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source

Push-Location -LiteralPath $projectRoot
try {
    # Seed 2021 is already complete. Never overwrite or silently resume a run.
    foreach ($seed in @(2022, 2023)) {
        $destination = "results_coursework/run_jinqi_baseline/bike_outflow/seed_$seed"
        if (Test-Path -LiteralPath $destination) {
            throw "Output already exists: $destination. Check it before rerunning; nothing was overwritten."
        }

        Write-Host "Running BIKE-Outflow baseline, seed $seed"
        & $pythonExe -u research/run_coursework.py --dataset bike_outflow --seed $seed --stage baseline --output $destination
        if ($LASTEXITCODE -ne 0) {
            throw "BIKE-Outflow seed $seed failed (exit $LASTEXITCODE). Queue stopped; output preserved."
        }

        $report = Get-Content -LiteralPath (Join-Path $destination 'report.json') -Raw | ConvertFrom-Json
        if ($report.configuration.dataset -ne 'bike_outflow' -or
            $report.configuration.seed -ne $seed -or
            $report.configuration.smoke -ne $false -or
            $null -eq $report.stages.baseline.test.rmse) {
            throw "Missing or mismatched baseline result: $destination. Queue stopped."
        }
        Write-Host "Completed seed $seed; test RMSE: $($report.stages.baseline.test.rmse)"
    }
    Write-Host 'BIKE-Outflow baseline seeds 2022 and 2023 completed.'
}
finally {
    Pop-Location
}
