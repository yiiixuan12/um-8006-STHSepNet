param(
    [string]$PythonPath = (Join-Path $PSScriptRoot '../../.envs/sthpilot-cu128/python.exe'),
    [string]$ResultsPath = (Join-Path $PSScriptRoot '../results_coursework/main_v1')
)
$ErrorActionPreference = 'Stop'
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonExe = [IO.Path]::GetFullPath($PythonPath)
$results = [IO.Path]::GetFullPath($ResultsPath)
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) { throw "Missing Python: $pythonExe" }
New-Item -ItemType Directory -Path $results -Force | Out-Null
Set-Location -LiteralPath $project
# Run all six original baselines first, then their paired periodic adapters.
# A failed process stops the queue; no failure is treated as a completed result.
foreach ($stage in @('baseline', 'improved')) {
    foreach ($seed in @(2021, 2022, 2023)) {
        foreach ($dataset in @('pems03', 'metr_la')) {
            $destination = Join-Path $results "$dataset/seed_$seed"
            $reportPath = Join-Path $destination 'report.json'
            $arguments = @('-u', 'research/run_coursework.py', '--dataset', $dataset,
                '--seed', "$seed", '--stage', $stage, '--output', $destination,
                '--batch-size', '4', '--target-chunk', '64')
            if (Test-Path -LiteralPath $reportPath) {
                $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
                if ($null -ne $report.stages.$stage) { continue }
                $last = Join-Path $destination "${stage}_last.pt"
                if ((Test-Path -LiteralPath $last) -or $stage -eq 'baseline') { $arguments += '--resume' }
            }
            $tag = Get-Date -Format 'yyyyMMdd_HHmmss'
            $log = Join-Path $results "${dataset}_${seed}_${stage}_${tag}.log"
            $event = @{ stage=$stage; dataset=$dataset; seed=$seed; output=$destination; log=$log;
                queue_pid=$PID; started_at=(Get-Date).ToString('o') } | ConvertTo-Json
            $event | Set-Content -LiteralPath (Join-Path $results 'queue_current.json') -Encoding UTF8
            Write-Output $event
            $quotedArguments = @($arguments | ForEach-Object { '"' + $_ + '"' })
            $child = Start-Process -FilePath $pythonExe -ArgumentList $quotedArguments -WorkingDirectory $project `
                -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.stderr" -Wait -PassThru
            if ($child.ExitCode -ne 0) { throw "Training failed: $dataset seed $seed $stage. Preserved $destination" }
            $finished = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
            if ($null -eq $finished.stages.$stage) { throw "Missing completed stage: $dataset $seed $stage" }
        }
    }
}
& $pythonExe research/summarize_coursework.py --results $results
if ($LASTEXITCODE -ne 0) { throw 'Aggregation failed' }
@{ status='completed'; ended_at=(Get-Date).ToString('o') } | ConvertTo-Json |
    Set-Content -LiteralPath (Join-Path $results 'queue_current.json') -Encoding UTF8
