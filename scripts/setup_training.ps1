param([string]$PythonPath = 'python', [switch]$PlanOnly)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'h1_common.ps1')
Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments @(
    '-c', "import sys,torch; assert sys.version_info >= (3,10), 'Python 3.10+ required'; assert torch.cuda.is_available(), 'Install CUDA-enabled PyTorch first'; print(sys.executable, torch.__version__, torch.cuda.get_device_name(0))")
Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments @(
    '-m', 'pip', 'install', '-r', (Resolve-H1Path 'scripts/requirements-training.txt'))
Invoke-H1Python -PythonPath $PythonPath -PlanOnly:$PlanOnly -Arguments @('-m', 'pip', 'check')
