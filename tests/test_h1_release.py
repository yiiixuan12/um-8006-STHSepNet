"""CPU-only release-interface checks; no data, BERT download, or GPU required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from research import run_h1 as h1


class ReleaseInterfaceTests(unittest.TestCase):
    def args(self, *extra):
        return h1.parser().parse_args(["--dataset", "bike_inflow", "--output", "out", *extra])

    def test_dataset_sampling_and_fixed_protocol(self):
        self.assertEqual(len(h1.DATASETS), 5)
        self.assertEqual(h1.history_length("bike_inflow"), 240)
        self.assertEqual(h1.history_length("bike_outflow"), 240)
        for city in ("pems03", "metr_la", "bj500"):
            self.assertEqual(h1.history_length(city), 2064)
        args = self.args("--stage", "baseline")
        report = h1.fresh_report(args, {}, {})
        self.assertEqual(report["protocol"]["adapter_max_epochs"], 5)
        self.assertEqual(report["protocol"]["fresh_baseline_max_epochs"], 20)
        self.assertEqual(report["configuration"]["batch_size"], 4)
        self.assertEqual(report["configuration"]["target_chunk"], 64)
        self.assertTrue(report["protocol"]["extra_information_disclosed"])

    def test_limits_do_not_silently_change_training(self):
        with self.assertRaisesRegex(ValueError, "evaluate-only"):
            h1.validate_arguments(self.args("--stage", "baseline", "--max-windows", "4"))
        h1.validate_arguments(self.args("--stage", "baseline", "--seed", "2022"))
        with self.assertRaisesRegex(ValueError, "--baseline"):
            h1.validate_arguments(self.args())
        with self.assertRaisesRegex(ValueError, "--checkpoint-dir"):
            h1.validate_arguments(self.args("--stage", "evaluate"))
        args = self.args("--stage", "evaluate", "--checkpoint-dir", "source", "--max-windows", "4")
        h1.validate_arguments(args)

    def test_relocates_only_model_paths(self):
        original = {"node_num": 295, "llm_layers": 12, "root_path": "D:/old/Bike",
                    "llm_path": "D:/old/bert", "d_ff": 32, "min_values": [0, 1]}
        specs = {"bike_inflow": {"nodes": 295, "folder": "Bike", "adj": "adj.csv"}}
        result = h1.relocation_config(original, "bike_inflow", Path("new_data"), Path("new_bert"), specs)
        self.assertEqual(result["root_path"], str((Path("new_data") / "Bike").resolve()))
        self.assertEqual(result["d_ff"], 32)
        self.assertEqual(result["min_values"], [0, 1])
        self.assertEqual(original["root_path"], "D:/old/Bike")
        with self.assertRaisesRegex(ValueError, "12-layer"):
            h1.relocation_config({**original, "llm_layers": 6}, "bike_inflow", Path("x"), Path("y"), specs)

    def test_audit_copy_and_baseline_checks(self):
        audit = {"rows": 100, "nodes": 295, "csv_bytes": 123, "csv_mtime_ns": 1}
        h1.check_audit(audit, {**audit, "csv_mtime_ns": 2, "window_counts": {}})
        with self.assertRaisesRegex(ValueError, "chronology"):
            h1.check_audit(audit, {**audit, "rows": 99})
        report = {"configuration": {"dataset": "bike_inflow", "seed": 2021, "smoke": False},
                  "stages": {"baseline": {"status": "completed_training", "test": {"mae": 1}}}}
        h1.check_baseline_record(report, "bike_inflow")
        bad = copy.deepcopy(report)
        bad["configuration"]["seed"] = 2020
        with self.assertRaisesRegex(ValueError, "supported seed"):
            h1.check_baseline_record(bad, "bike_inflow")
        paired = copy.deepcopy(report)
        paired["configuration"].update(seed=2023, baseline_seed=2022)
        self.assertEqual(h1.baseline_seed(paired), 2022)

    def test_checkpoint_paths_bom_and_rng_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "report.json").write_text(json.dumps({"x": 1}), encoding="utf-8-sig")
            self.assertEqual(h1.read_json(path / "report.json"), {"x": 1})
            with self.assertRaises(FileNotFoundError):
                h1.checkpoint_path(path, "improved")
            (path / "improved_best.pt").touch()
            self.assertEqual(h1.checkpoint_path(path, "improved"), path / "improved_best.pt")
        with self.assertRaisesRegex(ValueError, "missing RNG"):
            h1.restore_rng({}, None, None)

    def test_ablation_and_case_options_do_not_change_scientific_scope(self):
        h1.validate_arguments(self.args("--baseline", "base", "--arm", "global_long"))
        with self.assertRaisesRegex(ValueError, "BIKE-Inflow"):
            h1.validate_arguments(h1.parser().parse_args(["--dataset", "pems03", "--stage", "improved",
                "--baseline", "base", "--output", "out", "--arm", "no_utility"]))
        args = self.args("--stage", "evaluate", "--checkpoint-dir", "source", "--save-predictions",
                         "--origins", "7032", "7872", "8736")
        h1.validate_arguments(args)
        self.assertEqual(args.origins, [7032, 7872, 8736])
        args.max_windows = 4
        with self.assertRaisesRegex(ValueError, "cannot combine"):
            h1.validate_arguments(args)
        self.assertEqual(h1.report_arm({"configuration": {"arm": "global_long"}}), "global_long")
        self.assertEqual(h1.report_arm({"protocol": {"ablation": "no_utility"}}), "no_utility")
        self.assertEqual(h1.report_arm({}), "full")

    def test_resume_requires_full_state_and_original_budget(self):
        with self.assertRaisesRegex(ValueError, "complete optimizer/RNG"):
            h1.check_resume_state({"epoch": 3, "parameters": {}, "buffers": {}}, "improved")
        state = dict(epoch=3, parameters={}, buffers={}, optimizer={}, log=[1, 2, 3], stale=0,
                     best_value=1, best_epoch=3, rng=None, cuda_rng=None, numpy_rng=None, python_rng=None)
        h1.check_resume_state(state, "improved")
        h1.check_resume_state(state, "improved", {"epochs": 3, "training_log": [1, 2, 3]})
        with self.assertRaisesRegex(ValueError, "does not match"):
            h1.check_resume_state(state, "improved", {"epochs": 3, "training_log": [3, 2, 1]})
        with self.assertRaisesRegex(ValueError, "epoch budget"):
            h1.check_resume_state({**state, "epoch": 6, "log": list(range(6))}, "improved")

    def test_resume_detects_changed_scientific_source_without_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            file = source / "model.py"
            file.write_text("version_one", encoding="utf-8")
            output = root / "run"
            h1.snapshot_sources(output, source, ("model.py",))
            h1.check_source_snapshot(output, source, ("model.py",))
            file.write_text("version_two", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Scientific source changed"):
                h1.check_source_snapshot(output, source, ("model.py",))

    def test_same_seed_wrong_baseline_is_rejected_by_exact_tensors(self):
        expected = {"parameters": {"weight": [1, 2]}, "buffers": {"adj": [0, 1]}}
        same = copy.deepcopy(expected)
        h1.assert_same_compact(expected, same, lambda left, right: left == right)
        wrong = copy.deepcopy(same)
        wrong["parameters"]["weight"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "checkpoint tensors differ"):
            h1.assert_same_compact(expected, wrong, lambda left, right: left == right)
        wrong = copy.deepcopy(same)
        wrong["buffers"]["adj"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "checkpoint tensors differ"):
            h1.assert_same_compact(expected, wrong, lambda left, right: left == right)
        self.assertTrue(h1.same_model_config({"root_path": "a", "llm_path": "x", "d_ff": 32},
                                             {"root_path": "b", "llm_path": "y", "d_ff": 32}))
        self.assertFalse(h1.same_model_config({"d_ff": 32}, {"d_ff": 128}))


if __name__ == "__main__":
    unittest.main()
