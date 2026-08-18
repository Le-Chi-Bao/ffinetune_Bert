"""Stage-12 fixture tests: protocol, locking, run matrix, aggregation, and guards.

Uses a synthetic training entrypoint and synthetic metrics only. No AG News
research data and no official test split is touched.
"""
from __future__ import annotations

import json
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_stage12_protocol import ProtocolError, validate_protocol
from stage12_statistics import paired_differences, payload_sha256, summarize, summarize_paired

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 2]
CLASSES = ("World", "Sports", "Business", "Sci/Tech")

FAKE_TRAIN = '''import json, sys
from pathlib import Path
argv = sys.argv
def value(flag):
    return argv[argv.index(flag) + 1]
seed = int(value("--seed"))
run = Path(value("--output-dir")) / value("--run-name")
if value("--model-type") != "bert_baseline" and seed == 99:
    raise SystemExit(3)
for sub in ("checkpoints", "metrics", "logs"):
    (run / sub).mkdir(parents=True, exist_ok=True)
resolved = json.loads((run / "resolved_config.json").read_text())
base = 0.90 if value("--model-type") == "bert_baseline" else 0.92
score = base + seed * 0.001
metrics = {
    "loss": 0.30 - seed * 0.001, "accuracy": score, "precision_macro": score,
    "recall_macro": score, "f1_macro": score, "f1_weighted": score,
    "per_class": {c: {"f1": score, "precision": score, "recall": score, "support": 10}
                  for c in ["World", "Sports", "Business", "Sci/Tech"]},
}
(run / "metrics" / "best_validation_metrics.json").write_text(json.dumps(metrics))
(run / "checkpoints" / "best.pt").write_text(f"best-{value('--model-type')}-{seed}")
(run / "checkpoints" / "last.pt").write_text(f"last-{value('--model-type')}-{seed}")
(run / "data_signature.json").write_text(json.dumps({"protocol": "fixture", "checksum": "abc"}))
(run / "environment.json").write_text(json.dumps({"gpu_name": "fixture-gpu", "pytorch_version": "x",
    "transformers_version": "y", "pytorch_cuda_version": "z", "python_version": "p", "git_commit": None}))
(run / "summary.json").write_text(json.dumps({
    "seed": seed, "official_test_loaded": False, "official_test_evaluated": False,
    "best_epoch": 1 + seed, "best_validation_loss": metrics["loss"],
    "best_validation_accuracy": score, "best_validation_macro_f1": score,
    "total_parameters": 1000, "trainable_parameters": 1000,
    "peak_vram_mb": 100.0 + seed, "training_time_seconds": 10.0 + seed}))
(run / "history.csv").write_text("epoch\\n1\\n")
'''


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_failure(callable_, message: str) -> None:
    try:
        callable_()
    except Exception:
        return
    raise AssertionError(message)


def make_fixture(directory: Path, seeds: list[int] | None = None) -> dict[str, Any]:
    """Create a self-contained valid Stage-12 protocol fixture."""
    data = directory / "data"
    (data / "processed").mkdir(parents=True)
    (data / "reports").mkdir(parents=True)
    (data / "processed" / "research_train.parquet").write_text("train-fixture")
    (data / "processed" / "research_validation.parquet").write_text("validation-fixture")
    (data / "reports" / "final_data_report.json").write_text(json.dumps({
        "overall_status": "PASS", "READY_FOR_OFFICIAL_TRAINING": True,
        "label_mapping": {str(i): c for i, c in enumerate(CLASSES)}}))
    shared = {
        "model_name": "bert-base-uncased", "num_classes": 4, "max_length": 128, "epochs": 1,
        "train_batch_size": 8, "eval_batch_size": 16, "gradient_accumulation_steps": 4,
        "backbone_learning_rate": 2e-5, "head_learning_rate": 1e-4, "weight_decay": 0.01,
        "warmup_ratio": 0.1, "max_grad_norm": 1.0, "mixed_precision": "no",
        "early_stopping_patience": 2, "num_workers": 0, "dropout": 0.1,
    }
    baseline = directory / "baseline_config.json"
    baseline.write_text(json.dumps({**shared, "training_regime": "finetune"}))
    locked = directory / "locked_stage12_config.json"
    locked.write_text(json.dumps({
        **shared, "model_type": "ldtf_ablation", "training_regime": "finetune",
        "ablation_variant": "A0_full", "token_router_dim": 256, "depth_router_dim": 256,
        "scorer_type": "shared", "exclude_special_tokens": False}))
    config = {
        "stage": 12, "protocol_name": "fixture_v1", "seeds": list(seeds or SEEDS),
        "train_path": str(data / "processed" / "research_train.parquet"),
        "validation_path": str(data / "processed" / "research_validation.parquet"),
        "data_quality_report": str(data / "reports" / "final_data_report.json"),
        "stage11_locked_config": str(locked), "output_root": str(directory / "outputs"),
        "models": [
            {"experiment_name": "bert_baseline", "model_type": "bert_baseline", "config_source": str(baseline)},
            {"experiment_name": "ldtf_selected", "model_type": "from_stage11", "config_source": str(locked)},
        ],
        "selection_metric": "validation_macro_f1", "official_test_allowed": False,
    }
    config_path = directory / "stage12.json"
    config_path.write_text(json.dumps(config, indent=2))
    entrypoint = directory / "fake_train.py"
    entrypoint.write_text(FAKE_TRAIN)
    return {"config_path": config_path, "config": config, "entrypoint": entrypoint, "root": directory / "outputs"}


def run_runner(fixture: dict[str, Any], *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_stage12_multiseed", "--config", str(fixture["config_path"]),
         "--output-root", str(fixture["root"]), "--train-entrypoint", str(fixture["entrypoint"]), *flags],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_statistics() -> None:
    values = [0.90, 0.92, 0.94]
    stats = summarize(values)
    check(abs(stats["mean"] - 0.92) < 1e-12, "mean incorrect")
    check(abs(stats["std"] - statistics.stdev(values)) < 1e-12, "std must use ddof=1")
    check(stats["std_ddof"] == 1, "ddof must be recorded as 1")
    check(stats["min"] == 0.90 and stats["max"] == 0.94 and stats["median"] == 0.92, "min/max/median incorrect")
    check(summarize([0.5])["std"] is None, "single observation must report std=None, never 0.0")
    paired = paired_differences({0: 0.90, 1: 0.92}, {0: 0.93, 1: 0.91})
    check(abs(paired[0]["delta"] - 0.03) < 1e-12 and abs(paired[1]["delta"] + 0.01) < 1e-12, "paired delta sign wrong")
    aggregate = summarize_paired(paired)
    check(aggregate["number_positive"] == 1 and aggregate["number_negative"] == 1, "paired counts wrong")
    expect_failure(lambda: paired_differences({0: 1.0, 1: 1.0, 2: 1.0}, {0: 1.0, 1: 1.0}), "missing seed must be rejected")
    expect_failure(lambda: summarize([]), "empty metric list must be rejected")
    expect_failure(lambda: summarize([0.9, float("nan")]), "NaN metric must be rejected")
    expect_failure(lambda: summarize([0.9, float("inf")]), "Inf metric must be rejected")
    expect_failure(lambda: summarize([0.9, "0.91"]), "non-numeric metric must be rejected")
    expect_failure(lambda: paired_differences({0: float("nan")}, {0: 0.9}), "NaN in paired input must be rejected")
    print("PASS statistics (ddof=1, paired deltas, missing-seed guard, finite-value validation)")


def test_config_and_protocol() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = make_fixture(Path(directory))
        resolved = validate_protocol(fixture["config_path"])
        check(resolved["seeds"] == SEEDS, "seed list parsed incorrectly")
        check([m["experiment_name"] for m in resolved["models"]] == ["bert_baseline", "ldtf_selected"], "model list wrong")
        check(resolved["models"][1]["model_type"] == "ldtf_ablation", "from_stage11 must resolve to the locked model type")

        def mutate(**changes: Any) -> Path:
            payload = {**fixture["config"], **changes}
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(payload))
            return path

        for label, path in (
            ("duplicate seeds", mutate(seeds=[0, 0, 1])),
            ("empty seeds", mutate(seeds=[])),
            ("non-integer seeds", mutate(seeds=["a"])),
            ("test path", mutate(test_path="data/test.parquet")),
            ("official test allowed", mutate(official_test_allowed=True)),
            ("placeholder path", mutate(stage11_locked_config="outputs/<run>/locked.json")),
            ("wrong metric", mutate(selection_metric="accuracy")),
        ):
            expect_failure(lambda p=path: validate_protocol(p), f"validator accepted invalid protocol: {label}")

        report = Path(fixture["config"]["data_quality_report"])
        original = report.read_text()
        report.write_text(json.dumps({"overall_status": "FAIL", "READY_FOR_OFFICIAL_TRAINING": False}))
        expect_failure(lambda: validate_protocol(fixture["config_path"]), "validator accepted a failed data-quality gate")
        report.write_text(original)
        print("PASS protocol validator (valid accepted, 8 invalid protocols rejected)")


def test_runner() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = make_fixture(Path(directory))
        result = run_runner(fixture, "--continue-on-error")
        check(result.returncode == 0, f"runner failed: {result.stderr}")
        registry = json.loads((fixture["root"] / "run_registry.json").read_text())
        runs = registry["runs"]
        check(len(runs) == 6, f"run matrix must be 2 models x 3 seeds, got {len(runs)}")
        check(len({(r['experiment_name'], r['seed']) for r in runs}) == 6, "duplicate or missing run in matrix")
        check(all(r["status"] == "PASS" for r in runs), "all fixture runs should PASS")
        for run in runs:
            resolved = json.loads((Path(run["run_directory"]) / "resolved_config.json").read_text())
            check(resolved["seed"] == run["seed"], "seed not propagated to resolved config")
            check(Path(run["run_directory"]).name == f"seed_{run['seed']}", "run directory seed mismatch")
            check(resolved["protocol_hash"] == registry["protocol_hash"], "protocol hash not propagated")
            check(resolved["official_test_allowed"] is False, "resolved config must forbid official test")
        ldtf = json.loads((fixture["root"] / "ldtf_selected" / "seed_0" / "resolved_config.json").read_text())
        check(ldtf["model_type"] == "ldtf_ablation" and ldtf["ablation_variant"] == "A0_full", "locked architecture not used")
        check(ldtf["training_regime"] == "finetune", "locked training regime not used")

        lock = json.loads((fixture["root"] / "protocol_lock.json").read_text())
        check(lock["protocol_hash"] and lock["official_test_allowed"] is False, "protocol lock malformed")
        rerun = run_runner(fixture, "--skip-completed")
        check(rerun.returncode == 0, "skip-completed run failed")
        registry2 = json.loads((fixture["root"] / "run_registry.json").read_text())
        check(all(r["status"] == "SKIPPED_COMPLETED" for r in registry2["runs"]), "completed runs must be skipped")
        check(lock == json.loads((fixture["root"] / "protocol_lock.json").read_text()), "protocol lock was rewritten")

        # Same config must reproduce the same hash; changed config must be rejected.
        second_lock_check = run_runner(fixture, "--skip-completed")
        check(second_lock_check.returncode == 0, "stable protocol should not be rejected")
        original_config_bytes = fixture["config_path"].read_text()
        config = dict(fixture["config"])
        config["protocol_name"] = "fixture_v2"
        fixture["config_path"].write_text(json.dumps(config))
        mismatched = run_runner(fixture, "--skip-completed")
        check(mismatched.returncode != 0 and "Protocol lock mismatch" in (mismatched.stderr + mismatched.stdout),
              "runner must reject a changed protocol against an existing lock")
        # Restore the exact original bytes; the lock hashes file content, not parsed JSON.
        fixture["config_path"].write_text(original_config_bytes)
        check(run_runner(fixture, "--skip-completed").returncode == 0, "restored protocol should be accepted again")

        # Resume detection: incomplete run with last.pt only.
        target = fixture["root"] / "ldtf_selected" / "seed_1"
        (target / "summary.json").unlink()
        check((target / "checkpoints" / "last.pt").is_file(), "fixture must retain last.pt")
        without_resume = run_runner(fixture, "--skip-completed")
        check(without_resume.returncode == 0, f"runner failed without resume flag: {without_resume.stderr}")
        registry3 = json.loads((fixture["root"] / "run_registry.json").read_text())
        entry = next(r for r in registry3["runs"] if r["experiment_name"] == "ldtf_selected" and r["seed"] == 1)
        check(entry["status"] == "RESUME_REQUIRED", f"incomplete run must require resume, got {entry['status']}")
        with_resume = run_runner(fixture, "--skip-completed", "--resume-incomplete")
        check(with_resume.returncode == 0, "resume run failed")
        registry4 = json.loads((fixture["root"] / "run_registry.json").read_text())
        entry = next(r for r in registry4["runs"] if r["experiment_name"] == "ldtf_selected" and r["seed"] == 1)
        check(entry.get("resumed") is True and entry["status"] == "PASS", "resume was not detected/applied")
        print("PASS runner (matrix, seed propagation, lock, skip-completed, resume)")


def test_runner_failures() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = make_fixture(Path(directory), seeds=[0, 99, 2])
        result = run_runner(fixture, "--continue-on-error")
        check(result.returncode == 0, "continue-on-error should not abort the runner")
        registry = json.loads((fixture["root"] / "run_registry.json").read_text())
        failed = [r for r in registry["runs"] if r["status"] == "FAIL"]
        check(len(failed) == 1 and failed[0]["seed"] == 99, "failing seed must be recorded as FAIL")
        after = [r for r in registry["runs"] if r["experiment_name"] == "ldtf_selected" and r["seed"] == 2]
        check(after and after[0]["status"] == "PASS", "subsequent seed must still run after a failure")
        failures = json.loads((fixture["root"] / "failures.json").read_text())
        check(len(failures) == 1 and failures[0]["seed"] == 99, "failures.json must record the failed seed")
        seeds_present = {r["seed"] for r in registry["runs"]}
        check(seeds_present == {0, 99, 2}, "failed seed must not be replaced by a new seed")
        print("PASS runner failure handling (FAIL recorded, no seed substitution)")


def test_completion_requires_artifacts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = make_fixture(Path(directory))
        check(run_runner(fixture, "--continue-on-error").returncode == 0, "initial run failed")
        target = fixture["root"] / "bert_baseline" / "seed_0"
        (target / "checkpoints" / "best.pt").unlink()
        rerun = run_runner(fixture, "--skip-completed", "--resume-incomplete")
        registry = json.loads((fixture["root"] / "run_registry.json").read_text())
        entry = next(r for r in registry["runs"] if r["experiment_name"] == "bert_baseline" and r["seed"] == 0)
        check(entry["status"] != "SKIPPED_COMPLETED", "run without best.pt must not count as completed")
        print("PASS completion detection (missing artifacts are not PASS)")


def run_aggregator(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.aggregate_stage12_multiseed", "--root", str(root),
         "--baseline", "bert_baseline", "--candidate", "ldtf_selected",
         "--output-csv", str(root / "multiseed_results.csv"), "--output-json", str(root / "multiseed_results.json")],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_aggregator() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = make_fixture(Path(directory))
        check(run_runner(fixture, "--continue-on-error").returncode == 0, "fixture runs failed")
        root = fixture["root"]
        result = run_aggregator(root)
        check(result.returncode == 0, f"aggregator failed: {result.stderr}")
        payload = json.loads((root / "multiseed_results.json").read_text())
        baseline_scores = [0.900, 0.901, 0.902]
        candidate_scores = [0.920, 0.921, 0.922]
        baseline_stats = payload["models"]["bert_baseline"]["metrics"]["f1_macro"]
        check(abs(baseline_stats["mean"] - statistics.fmean(baseline_scores)) < 1e-9, "baseline mean incorrect")
        check(abs(baseline_stats["std"] - statistics.stdev(baseline_scores)) < 1e-9, "baseline std must use ddof=1")
        check(abs(baseline_stats["median"] - 0.901) < 1e-9, "median incorrect")
        check(abs(baseline_stats["min"] - 0.900) < 1e-9 and abs(baseline_stats["max"] - 0.902) < 1e-9, "min/max incorrect")
        candidate_stats = payload["models"]["ldtf_selected"]["metrics"]["f1_macro"]
        check(abs(candidate_stats["mean"] - statistics.fmean(candidate_scores)) < 1e-9, "candidate mean incorrect")
        summary = payload["paired_comparison"]["summary"]
        check(abs(summary["mean_paired_delta"] - 0.02) < 1e-9, "paired delta mean incorrect")
        check(summary["number_positive"] == 3 and summary["number_negative"] == 0, "paired counts incorrect")
        for row in payload["paired_comparison"]["per_seed"]:
            check(abs(row["delta_macro_f1"] - (row["ldtf_macro_f1"] - row["baseline_macro_f1"])) < 1e-12, "delta sign inverted")
        per_class = payload["models"]["ldtf_selected"]["per_class_f1"]
        check(set(per_class) == set(CLASSES), "per-class aggregate missing classes")
        efficiency = payload["models"]["bert_baseline"]["efficiency"]
        check(abs(efficiency["peak_vram_mb"]["mean"] - 101.0) < 1e-9, "VRAM mean incorrect")
        check(abs(efficiency["training_time_seconds"]["mean"] - 11.0) < 1e-9, "training-time mean incorrect")
        check("significance is not claimed" in payload["statistical_claim"], "missing statistical caveat")
        for name in ("multiseed_results.csv", "per_class_multiseed.csv", "efficiency_multiseed.csv", "paired_seed_differences.csv"):
            check((root / name).is_file(), f"missing aggregate output {name}")

        manifest = json.loads((root / "locked_stage13_manifest.json").read_text())
        check(set(manifest["models"]) == {"bert_baseline", "ldtf_selected"}, "manifest missing a model")
        for entries in manifest["models"].values():
            check([e["seed"] for e in entries] == SEEDS, "manifest must contain every locked seed")
            check(all(e["checkpoint_sha256"] for e in entries), "manifest checkpoints need checksums")
        core = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        check(payload_sha256(core) == manifest["manifest_sha256"], "manifest checksum mismatch")
        print("PASS aggregator (mean/std/median, paired deltas, per-class, efficiency, manifest)")


def test_aggregator_rejections() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = make_fixture(Path(directory))
        check(run_runner(fixture, "--continue-on-error").returncode == 0, "fixture runs failed")
        root = fixture["root"]
        check(run_aggregator(root).returncode == 0, "baseline aggregation should pass")
        backup = Path(directory) / "backup"
        shutil.copytree(root, backup)

        def restore() -> None:
            shutil.rmtree(root)
            shutil.copytree(backup, root)

        cases: list[tuple[str, Any]] = [
            ("official test evaluated", (root / "ldtf_selected" / "seed_0" / "summary.json", lambda d: {**d, "official_test_evaluated": True})),
            ("official test loaded", (root / "ldtf_selected" / "seed_1" / "summary.json", lambda d: {**d, "official_test_loaded": True})),
            ("parameter mismatch", (root / "ldtf_selected" / "seed_2" / "summary.json", lambda d: {**d, "total_parameters": 999})),
            ("config drift", (root / "ldtf_selected" / "seed_1" / "resolved_config.json", lambda d: {**d, "head_learning_rate": 0.5})),
            ("data signature drift", (root / "ldtf_selected" / "seed_0" / "data_signature.json", lambda d: {**d, "checksum": "different"})),
            ("protocol hash drift", (root / "bert_baseline" / "seed_0" / "resolved_config.json", lambda d: {**d, "protocol_hash": "tampered"})),
        ]
        for label, (path, mutate) in cases:
            payload = json.loads(path.read_text())
            path.write_text(json.dumps(mutate(payload)))
            result = run_aggregator(root)
            check(result.returncode != 0, f"aggregator accepted invalid state: {label}")
            restore()

        # Missing seed must be rejected, never silently dropped.
        registry = json.loads((root / "run_registry.json").read_text())
        registry["runs"] = [r for r in registry["runs"] if not (r["experiment_name"] == "ldtf_selected" and r["seed"] == 2)]
        (root / "run_registry.json").write_text(json.dumps(registry))
        result = run_aggregator(root)
        check(result.returncode != 0, "aggregator accepted a missing paired seed")
        restore()

        # A FAIL status must block aggregation rather than being skipped.
        registry = json.loads((root / "run_registry.json").read_text())
        registry["runs"][0]["status"] = "FAIL"
        (root / "run_registry.json").write_text(json.dumps(registry))
        check(run_aggregator(root).returncode != 0, "aggregator accepted a FAILed run")
        restore()
        print("PASS aggregator rejections (test guard, param/config/signature drift, missing seed)")


def test_environment_warning() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = make_fixture(Path(directory))
        check(run_runner(fixture, "--continue-on-error").returncode == 0, "fixture runs failed")
        root = fixture["root"]
        path = root / "ldtf_selected" / "seed_0" / "environment.json"
        payload = json.loads(path.read_text())
        payload["gpu_name"] = "different-gpu"
        path.write_text(json.dumps(payload))
        result = run_aggregator(root)
        check(result.returncode == 0, "environment mismatch should warn, not abort accuracy aggregation")
        aggregated = json.loads((root / "multiseed_results.json").read_text())
        check(aggregated["same_environment"] is False, "environment mismatch must be flagged")
        check("multiple environments" in result.stderr, "environment mismatch must emit a warning")
        print("PASS environment mismatch warning (flagged, efficiency not treated as comparable)")


def test_same_seed_repeatability() -> None:
    """Same seed and deterministic CPU setup must reproduce identical draws."""
    def draw(seed: int) -> list[float]:
        import numpy as np
        import torch
        from training_utils import set_seed

        set_seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return [
            random.random(), float(np.random.rand()),
            float(torch.rand(1).item()), float(torch.rand(4, generator=generator).sum()),
        ]

    first, second, other = draw(0), draw(0), draw(1)
    check(first == second, f"same seed must reproduce identical draws: {first} vs {second}")
    check(first != other, "different seeds must not produce identical draws")
    print("PASS same-seed repeatability (deterministic CPU fixture, exact match)")


def main() -> None:
    test_statistics()
    test_config_and_protocol()
    test_runner()
    test_runner_failures()
    test_completion_requires_artifacts()
    test_aggregator()
    test_aggregator_rejections()
    test_environment_warning()
    test_same_seed_repeatability()
    print("All Stage-12 implementation tests PASSED. Official test was not loaded or evaluated.")


if __name__ == "__main__":
    main()
