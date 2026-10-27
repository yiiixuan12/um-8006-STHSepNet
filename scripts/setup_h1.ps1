param(
    [string]$PythonPath = 'python',
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'h1_common.ps1')

# Activate a separate Python 3.10 environment and install suitable PyTorch first.
# Checking it before pip avoids an implicit, unselected Torch installation.
Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments @(
    '-c', "import sys; print(sys.version); sys.exit(0 if sys.version_info[:2] == (3, 10) else 'H1 setup requires Python 3.10; use a separate environment.')"
)
Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments @(
    '-c', "import torch; print('Existing PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda)"
)
Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments @(
    '-m', 'pip', 'install', '-r', (Resolve-H1Path 'requirements-pilot.txt')
)
Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments @('-m', 'pip', 'check')
