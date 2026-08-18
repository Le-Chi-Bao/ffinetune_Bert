"""Stage-11 tests: ablation variants, aggregation, and locked Stage-12 selection.

No real training; all runs are synthetic fixture directories.
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

from models.ldtf_ablation import ABLATION_VARIANTS  # noqa: E402
from scripts.select_stage11_config import (  # noqa: E402
    SelectionError,
    build_locked_config,
    rank_rows,
    select_best,
)
from tests.fixtures import (  # noqa: E402
    make_run_config,
    make_stage10_run,
    make_stage11_ablation_run,
    write_json,
)

CONFIGURED_VARIANTS = [
    ("A0_full", "A0_full", 256, 256, False),
    ("A1_no_token_router", "A1_no_token_router", 256, 256, False),
    ("A2_no_depth_router", "A2_no_depth_router", 256, 256, False),
    ("A3_final_layer", "A3_final_layer", 256, 256, False),
    ("A4_shared_token_query", "A4_shared_token_query", 256, 256, False),
    ("A5_shared_depth_query", "A5_shared_depth_query", 256, 256, False),
    ("A6_class_specific_scorer", "A6_class_specific_scorer", 256, 256, False),
    ("B1_router_dim64", "A0_full", 64, 64, False),
    ("B2_router_dim128", "A0_full", 128, 128, False),
    ("B4_exclude_special_tokens", "A0_full", 256, 256, True),
]


def _row(name: str, macro_f1: float, loss: float, params: int = 109_681_921) -> dict:
    return {
        "Variant": name,
        "Macro F1": macro_f1,
        "Val loss": loss,
        "Trainable params": params,
        "Params": params,
    }


class TestAblationVariantCoverage(unittest.TestCase):
    def test_all_a_variants_are_implemented(self):
        for variant in (
            "A0_full", "A1_no_token_router", "A2_no_depth_router", "A3_final_layer",
            "A4_shared_token_query", "A5_shared_depth_query", "A6_class_specific_scorer",
        ):
            self.assertIn(variant, ABLATION_VARIANTS)

    def test_shipped_config_covers_a0_a6_b1_b2_b4(self):
        config = json.loads((ROOT / "configs" / "stage11_ablation.json").read_text())
        names = [variant["name"] for variant in config["variants"]]
        for expected in (
            "A0_full", "A1_no_token_router", "A2_no_depth_router", "A3_final_layer",
            "A4_shared_token_query", "A5_shared_depth_query", "A6_class_specific_scorer",
            "B1_router_dim64", "B2_router_dim128", "B4_exclude_special_tokens",
        ):
            self.assertIn(expected, names)
        self.assertEqual(len(names), 10)


class TestSelectionRanking(unittest.TestCase):
    def test_highest_macro_f1_is_selected(self):
        rows = [_row("A0_full", 0.930, 0.21), _row("A6_class_specific_scorer", 0.941, 0.19)]
        best, _ = select_best(rows)
        self.assertEqual(best["Variant"], "A6_class_specific_scorer")

    def test_tie_on_f1_broken_by_lower_loss(self):
        rows = [_row("A0_full", 0.940, 0.25), _row("B2_router_dim128", 0.940, 0.19)]
        best, _ = select_best(rows)
        self.assertEqual(best["Variant"], "B2_router_dim128")

    def test_tie_on_f1_and_loss_broken_by_fewer_parameters(self):
        rows = [
            _row("A0_full", 0.940, 0.20, params=109_681_921),
            _row("B1_router_dim64", 0.940, 0.20, params=100_000_000),
        ]
        best, _ = select_best(rows)
        self.assertEqual(best["Variant"], "B1_router_dim64")

    def test_ranking_is_deterministic_regardless_of_input_order(self):
        rows = [_row("A0_full", 0.93, 0.21), _row("A3_final_layer", 0.95, 0.18), _row("B1_router_dim64", 0.94, 0.19)]
        forward = [item["Variant"] for item in rank_rows(rows)]
        backward = [item["Variant"] for item in rank_rows(list(reversed(rows)))]
        self.assertEqual(forward, backward)
        self.assertEqual(forward[0], "A3_final_layer")

    def test_ranking_trace_flags_ties(self):
        rows = [_row("A0_full", 0.94, 0.20), _row("B1_router_dim64", 0.94, 0.21)]
        _, trace = select_best(rows)
        self.assertTrue(all(entry["tied_on_macro_f1_with_best"] for entry in trace))

    def test_empty_results_are_rejected(self):
        with self.assertRaises(SelectionError):
            select_best([])


class TestLockedConfigConstruction(unittest.TestCase):
    def _ablation_config(self):
        return {
            "base_model": "bert-base-uncased",
            "max_length": 128,
            "variants": [
                {
                    "name": name, "variant": variant,
                    "token_router_dim": token_dim, "depth_router_dim": depth_dim,
                    **({"exclude_special_tokens": True} if exclude else {}),
                }
                for name, variant, token_dim, depth_dim, exclude in CONFIGURED_VARIANTS
            ],
        }

    def _build(self, selected_name: str, regime: str = "finetune"):
        rows = [_row(name, 0.90, 0.30) for name, *_ in CONFIGURED_VARIANTS]
        for row in rows:
            if row["Variant"] == selected_name:
                row["Macro F1"] = 0.99
        best, trace = select_best(rows)
        return build_locked_config(
            best=best, trace=trace,
            aggregate={"comparison_protocol": {}, "data_signature": {}},
            ablation_config=self._ablation_config(), training_regime=regime,
            stage10_config=make_run_config(),
        )

    def test_a0_locks_the_full_ldtf_model(self):
        locked = self._build("A0_full")
        self.assertEqual(locked["model_type"], "ldtf")
        self.assertEqual(locked["ablation_variant"], "A0_full")
        self.assertEqual(locked["scorer_type"], "shared")

    def test_a6_locks_class_specific_scorer(self):
        locked = self._build("A6_class_specific_scorer")
        self.assertEqual(locked["model_type"], "ldtf_ablation")
        self.assertEqual(locked["scorer_type"], "class-specific")

    def test_b1_preserves_router_dimension(self):
        locked = self._build("B1_router_dim64")
        self.assertEqual(locked["token_router_dim"], 64)
        self.assertEqual(locked["depth_router_dim"], 64)
        self.assertEqual(locked["model_type"], "ldtf")

    def test_b4_preserves_exclude_special_tokens(self):
        locked = self._build("B4_exclude_special_tokens")
        self.assertTrue(locked["exclude_special_tokens"])

    def test_regime_is_inherited_from_stage10(self):
        self.assertEqual(self._build("A0_full", regime="frozen")["training_regime"], "frozen")

    def test_locked_config_is_hashed_and_test_free(self):
        locked = self._build("A0_full")
        self.assertEqual(len(locked["locked_config_sha256"]), 64)
        self.assertIs(locked["official_test_evaluated"], False)
        for key in locked:
            self.assertNotIn("test_path", key)


class TestSelectionCliRejections(unittest.TestCase):
    """The CLI must refuse to lock a config from incomplete or tainted inputs."""

    def _run(self, tmp: Path, *, rows, stage10_overrides=None, extra_args=()):
        aggregate = write_json({"results": rows, "comparison_protocol": {}, "data_signature": {}}, tmp / "agg.json")
        config = write_json(
            {
                "base_model": "bert-base-uncased", "max_length": 128,
                "variants": [
                    {"name": n, "variant": v, "token_router_dim": t, "depth_router_dim": d}
                    for n, v, t, d, _ in CONFIGURED_VARIANTS
                ],
            },
            tmp / "ablation.json",
        )
        make_stage10_run(tmp / "stage10")
        if stage10_overrides:
            summary = json.loads((tmp / "stage10" / "summary.json").read_text())
            summary.update(stage10_overrides)
            write_json(summary, tmp / "stage10" / "summary.json")
        return subprocess.run(
            [
                sys.executable, str(ROOT / "scripts" / "select_stage11_config.py"),
                "--aggregate-json", str(aggregate), "--ablation-config", str(config),
                "--stage10-summary", str(tmp / "stage10" / "summary.json"),
                "--output", str(tmp / "locked_stage12_config.json"), *extra_args,
            ],
            capture_output=True, text=True, cwd=ROOT,
        )

    def test_missing_variant_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_row(name, 0.9, 0.3) for name, *_ in CONFIGURED_VARIANTS[:-1]]
            result = self._run(Path(tmp), rows=rows)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing variants", result.stderr.lower())

    def test_stage10_run_that_touched_test_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_row(name, 0.9, 0.3) for name, *_ in CONFIGURED_VARIANTS]
            result = self._run(Path(tmp), rows=rows, stage10_overrides={"official_test_evaluated": True})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("official test", result.stderr.lower())

    def test_complete_inputs_produce_a_locked_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_row(name, 0.9, 0.3) for name, *_ in CONFIGURED_VARIANTS]
            rows[0]["Macro F1"] = 0.97
            result = self._run(Path(tmp), rows=rows)
            self.assertEqual(result.returncode, 0, result.stderr)
            locked = json.loads((Path(tmp) / "locked_stage12_config.json").read_text())
            self.assertEqual(locked["selected_run_name"], "A0_full")
            self.assertEqual(locked["training_regime"], "finetune")

    def test_existing_locked_config_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_row(name, 0.9, 0.3) for name, *_ in CONFIGURED_VARIANTS]
            self.assertEqual(self._run(Path(tmp), rows=rows).returncode, 0)
            second = self._run(Path(tmp), rows=rows)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("immutable", second.stderr.lower())


class TestStage11Aggregation(unittest.TestCase):
    def test_aggregator_rejects_incompatible_runs(self):
        """Runs with mismatched comparison protocols must never be averaged together."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "stage11"
            registry = []
            for index, (name, variant, token_dim, depth_dim, exclude) in enumerate(CONFIGURED_VARIANTS[:3]):
                run_dir = root / name
                make_stage11_ablation_run(
                    run_dir, name=name, variant=variant, macro_f1=0.90 + index * 0.01,
                    loss=0.30 - index * 0.01, token_router_dim=token_dim,
                    depth_router_dim=depth_dim, exclude_special_tokens=exclude,
                )
                registry.append({"name": name, "path": str(run_dir), "status": "PASS"})
            # Corrupt one run's seed so the comparison protocol no longer matches.
            config = json.loads((root / "A1_no_token_router" / "config.json").read_text())
            config["seed"] = 999
            write_json(config, root / "A1_no_token_router" / "config.json")
            write_json(registry, root / "run_registry.json")
            result = subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts" / "aggregate_stage11_ablation.py"),
                    "--root", str(root), "--output-csv", str(root / "out.csv"),
                    "--output-json", str(root / "out.json"),
                ],
                capture_output=True, text=True, cwd=ROOT,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("incompatible", result.stderr.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
