"""Stage-13 aggregation test using synthetic official-test result artifacts.

The result files are fabricated. The official AG News test split is never loaded.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

from stage12_statistics import payload_sha256  # noqa: E402
from tests.fixtures import make_metrics, write_json  # noqa: E402

SEEDS = [0, 1, 2]
MODELS = ("bert_baseline", "ldtf_selected")
N_EXAMPLES = 20


def build_stage13_results(root: Path) -> Path:
    """Fabricate a complete set of Stage-13 official-test artifacts."""
    manifest_core = {
        "stage": 13,
        "protocol_hash": "c" * 64,
        "data_signature_hash": "d" * 64,
        "official_test_evaluated": False,
        "seeds": list(SEEDS),
        "models": {name: [{"seed": s, "checkpoint": f"{name}_{s}.pt"} for s in SEEDS] for name in MODELS},
    }
    manifest_hash = payload_sha256(manifest_core)
    manifest = write_json({**manifest_core, "manifest_sha256": manifest_hash}, root / "manifest.json")

    labels = [index % 4 for index in range(N_EXAMPLES)]
    for name in MODELS:
        for seed in SEEDS:
            run_dir = root / name / f"seed_{seed}"
            # The candidate is deliberately a little stronger than the baseline.
            wrong = 4 if name == "bert_baseline" else 2
            predictions = [
                (label + 1) % 4 if index < wrong else label
                for index, label in enumerate(labels)
            ]
            accuracy = sum(p == l for p, l in zip(predictions, labels)) / N_EXAMPLES
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "test_predictions.csv").open("w", encoding="utf-8") as handle:
                handle.write("index,label,prediction\n")
                for index, (label, prediction) in enumerate(zip(labels, predictions)):
                    handle.write(f"{index},{label},{prediction}\n")
            write_json(make_metrics(macro_f1=accuracy, loss=0.2, accuracy=accuracy), run_dir / "test_metrics.json")
            write_json(
                {
                    "experiment_name": name, "seed": seed,
                    "manifest_sha256": manifest_hash, "protocol_hash": "c" * 64,
                    "official_test_evaluated": True,
                    "test_accuracy": accuracy, "test_macro_f1": accuracy,
                },
                run_dir / "test_summary.json",
            )
    write_json(
        {
            "manifest_sha256": manifest_hash, "protocol_hash": "c" * 64,
            "seeds": list(SEEDS), "official_test_evaluated": True,
        },
        root / "stage13_run_registry.json",
    )
    return manifest


class TestStage13Aggregation(unittest.TestCase):
    def _aggregate(self, root: Path, manifest: Path):
        return subprocess.run(
            [
                sys.executable, str(ROOT / "scripts" / "aggregate_stage13_official_test.py"),
                "--root", str(root), "--manifest", str(manifest),
                "--baseline", "bert_baseline", "--candidate", "ldtf_selected",
                "--output-csv", str(root / "summary.csv"),
                "--output-json", str(root / "summary.json"),
            ],
            capture_output=True, text=True, cwd=ROOT,
        )

    def test_aggregates_all_seeds_with_paired_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_stage13_results(root)
            result = self._aggregate(root, manifest)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "summary.json").read_text())
            self.assertEqual(payload["seeds"], SEEDS)
            for name in MODELS:
                self.assertEqual(payload["models"][name]["seeds"], SEEDS)
                self.assertEqual(payload["models"][name]["metrics"]["accuracy"]["std_ddof"], 1)
            paired = payload["paired_comparison"]["summary"]
            self.assertEqual(paired["n_seeds"], 3)
            self.assertGreater(paired["mean_paired_delta"], 0)
            self.assertIn("not claimed", paired["statistical_claim"].lower())

    def test_prediction_level_contingency_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_stage13_results(root)
            self.assertEqual(self._aggregate(root, manifest).returncode, 0)
            payload = json.loads((root / "summary.json").read_text())
            contingency = payload["paired_comparison"]["prediction_level_contingency"]
            self.assertEqual(sorted(int(k) for k in contingency), SEEDS)
            for table in contingency.values():
                self.assertEqual(table["n_examples"], N_EXAMPLES)
                self.assertEqual(table["only_candidate_correct"], 2)
                self.assertEqual(table["only_baseline_correct"], 0)

    def test_missing_seed_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_stage13_results(root)
            import shutil

            shutil.rmtree(root / "ldtf_selected" / "seed_2")
            result = self._aggregate(root, manifest)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("never be dropped", result.stderr.lower())

    def test_manifest_hash_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_stage13_results(root)
            registry = json.loads((root / "stage13_run_registry.json").read_text())
            registry["manifest_sha256"] = "f" * 64
            write_json(registry, root / "stage13_run_registry.json")
            result = self._aggregate(root, manifest)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest hash", result.stderr.lower())

    def test_selection_note_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_stage13_results(root)
            self.assertEqual(self._aggregate(root, manifest).returncode, 0)
            payload = json.loads((root / "summary.json").read_text())
            self.assertIn("no model, seed", payload["selection_note"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
