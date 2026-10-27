# Shared execution helpers only; scientific logic stays in research/run_h1.py.
$script:H1ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

function Resolve-H1Path {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $script:H1ProjectRoot $Path))
}

function Invoke-H1Python {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [switch]$PlanOnly
    )
    # Resource/output paths, including relative Python paths, are repo-relative.
    $python = $PythonPath
    if ($PythonPath.Contains('/') -or $PythonPath.Contains('\')) {
        $python = Resolve-H1Path $PythonPath
    }
    if ($PlanOnly) {
        [ordered]@{ python = $python; working_directory = $script:H1ProjectRoot;
            arguments = @($Arguments) } | ConvertTo-Json -Depth 4 -Compress
        return
    }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        $python = (Get-Command -Name $python -CommandType Application -ErrorAction Stop).Source
    }
    Push-Location -LiteralPath $script:H1ProjectRoot
    try {
        & $python @Arguments
        # PowerShell 5.1 does not throw automatically on native process failures.
        $nativeExit = $LASTEXITCODE
        if ($null -eq $nativeExit -or $nativeExit -ne 0) {
            throw "H1 Python command failed (exit $nativeExit). No later step was started."
        }
    }
    finally { Pop-Location }
}
