"""H1 PowerShell wrappers: argument, path and native-failure checks only.

No dependency installation, model construction, training or GPU calls.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research import run_h1 as h1

ROOT = Path(__file__).resolve().parents[1]
SHELL = shutil.which("powershell.exe") or shutil.which("pwsh")


@unittest.skipUnless(SHELL, "PowerShell is required to test Windows wrappers")
class H1ScriptTests(unittest.TestCase):
    def invoke(self, name, *arguments, cwd=None):
        return subprocess.run(
            [SHELL, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(ROOT / "scripts" / name), "-PythonPath", sys.executable, *arguments],
            cwd=cwd or ROOT, capture_output=True, text=True, errors="replace", timeout=30,
        )

    def plans(self, name, *arguments, cwd=None):
        result = self.invoke(name, *arguments, "-PlanOnly", cwd=cwd)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]

    def python_args(self, plan):
        self.assertEqual(Path(plan["working_directory"]).resolve(), ROOT)
        self.assertEqual(plan["python"], sys.executable)
        self.assertEqual(plan["arguments"][0], "research/run_h1.py")
        args = h1.parser().parse_args(plan["arguments"][1:])
        h1.validate_arguments(args)
        return args

    def test_baseline_five_datasets_resolve_paths_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for dataset in h1.DATASETS:
                with self.subTest(dataset=dataset):
                    plan, = self.plans("run_h1.ps1", "-Dataset", dataset, "-Stage", "baseline",
                                       "-Output", "runs/h1/plan only", cwd=directory)
                    args = self.python_args(plan)
                    self.assertEqual(args.output, ROOT / "runs/h1/plan only")
                    self.assertEqual(args.data_root, ROOT / "dataset")
                    self.assertEqual(args.bert_path, ROOT / "huggingface/google_bert")
                    self.assertEqual(args.seed, 2021)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_improved_preserves_seed_resume_and_paths_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="h1 script ") as directory:
            base, numeric, output = [str(Path(directory) / name) for name in ("base dir", "fit file.pt", "out dir")]
            plan, = self.plans("run_h1.ps1", "-Dataset", "bike_inflow", "-Stage", "improved",
                               "-Seed", "2023", "-Baseline", base, "-NumericCalibration", numeric,
                               "-Output", output, "-Resume")
            args = self.python_args(plan)
            self.assertEqual(str(args.baseline), base)
            self.assertEqual(str(args.numeric_calibration), numeric)
            self.assertEqual(str(args.output), output)
            self.assertEqual(args.seed, 2023)
            self.assertTrue(args.resume)
            self.assertFalse(Path(output).exists())

    def test_ablation_and_evaluation_are_forwarded_without_budget_changes(self):
        for arm in ("global_long", "no_utility"):
            plan, = self.plans("run_h1.ps1", "-Dataset", "bike_inflow", "-Stage", "improved",
                               "-Arm", arm, "-Baseline", "runs/h1/base", "-Output", f"runs/h1/{arm}")
            self.assertEqual(self.python_args(plan).arm, arm)
        plan, = self.plans("run_h1.ps1", "-Dataset", "metr_la", "-Stage", "evaluate",
                           "-CheckpointDir", "runs/h1/weights", "-Output", "runs/h1/eval",
                           "-EvalModel", "baseline", "-MaxWindows", "2", "-SavePredictions")
        args = self.python_args(plan)
        self.assertEqual(args.eval_model, "baseline")
        self.assertEqual(args.max_windows, 2)
        self.assertTrue(args.save_predictions)

    def test_invalid_requests_fail_before_launch(self):
        for extras in (
            ("-Dataset", "pems03", "-Stage", "improved", "-Output", "unused"),
            ("-Dataset", "pems03", "-Stage", "evaluate", "-Output", "unused"),
            ("-Dataset", "pems03", "-Stage", "improved", "-Arm", "global_long",
             "-Baseline", "base", "-Output", "unused"),
        ):
            with self.subTest(extras=extras):
                result = self.invoke("run_h1.ps1", *extras, "-PlanOnly")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout.strip(), "")

    def test_smoke_default_has_no_training_or_data_and_opt_in_check_is_cpu(self):
        plans = self.plans("smoke_h1.ps1")
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0]["arguments"], ["research/run_h1.py", "--help"])
        self.assertEqual(plans[1]["arguments"][:3], ["-m", "unittest", "discover"])
        plans = self.plans("smoke_h1.ps1", "-CheckData", "-Dataset", "bike_outflow")
        self.assertEqual(len(plans), 3)
        args = self.python_args(plans[2])
        self.assertTrue(args.check_data)
        self.assertEqual(args.stage, "baseline")

    def test_setup_and_result_steps_are_dry_run_and_sequential(self):
        plans = self.plans("setup_h1.ps1")
        self.assertEqual(len(plans), 4)
        self.assertEqual(plans[0]["arguments"][0], "-c")
        self.assertIn("(3, 10)", plans[0]["arguments"][1])
        self.assertIn("import torch", plans[1]["arguments"][1])
        self.assertEqual(plans[2]["arguments"][:4], ["-m", "pip", "install", "-r"])
        self.assertEqual(plans[3]["arguments"], ["-m", "pip", "check"])
        plans = self.plans("check_h1_results.ps1", "-Redraw")
        self.assertEqual(len(plans), 3)
        self.assertEqual(plans[0]["arguments"][0], "research/check_h1_results.py")
        self.assertIn("--output-dir", plans[1]["arguments"])
        self.assertIn(str(ROOT / "outputs/h1_cases"), plans[2]["arguments"])

    def test_native_arguments_and_failure_restore_callers_directory(self):
        def quote(value):
            return "'" + str(value).replace("'", "''") + "'"

        with tempfile.TemporaryDirectory(prefix="h1 caller ") as directory:
            payload = r"C:\node with spaces\历史.pkl"
            success = (
                "$ErrorActionPreference = 'Stop'; "
                f". {quote(ROOT / 'scripts/h1_common.ps1')}; Invoke-H1Python "
                f"-PythonPath {quote(sys.executable)} "
                "-Arguments @('-c', 'import json,sys; print(json.dumps(sys.argv[1:]))', "
                f"{quote(payload)})"
            )
            result = subprocess.run([SHELL, "-NoProfile", "-NonInteractive", "-Command", success],
                                    cwd=directory, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [payload])
            command = (
                "$ErrorActionPreference = 'Stop'; "
                f". {quote(ROOT / 'scripts/h1_common.ps1')}; "
                "try { Invoke-H1Python "
                f"-PythonPath {quote(sys.executable)} "
                "-Arguments @('-c', 'import sys; sys.exit(19)'); "
                "Write-Output 'UNEXPECTED_LATER_STEP'; exit 0 } "
                "catch { Write-Output ('restored=' + "
                f"((Get-Location).Path -eq {quote(directory)})); exit 23 }}"
            )
            result = subprocess.run([SHELL, "-NoProfile", "-NonInteractive", "-Command", command],
                                    cwd=directory, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertIn("restored=True", result.stdout)
            self.assertNotIn("UNEXPECTED_LATER_STEP", result.stdout)


if __name__ == "__main__":
    unittest.main()
