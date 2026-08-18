"""Stage-13 locked-manifest validator.

Refuses to let the official test split be opened unless the Stage-12 manifest is
internally consistent, its recorded hash still matches its contents, and every
referenced checkpoint is present with the exact SHA-256 recorded at lock time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage12_statistics import file_sha256, payload_sha256  # noqa: E402

MANIFEST_HASH_KEY = "manifest_sha256"


class ManifestError(RuntimeError):
    """Raised when the Stage-13 locked manifest cannot be trusted."""


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestError(f"Locked Stage-13 manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ManifestError(f"Manifest must be a JSON object: {manifest_path}")
    return manifest


def verify_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Recompute the manifest hash over its core payload and compare."""
    recorded = manifest.get(MANIFEST_HASH_KEY)
    if not recorded:
        raise ManifestError(f"Manifest is missing {MANIFEST_HASH_KEY!r}.")
    core = {key: value for key, value in manifest.items() if key != MANIFEST_HASH_KEY}
    recomputed = payload_sha256(core)
    if recomputed != recorded:
        raise ManifestError(
            "Manifest checksum mismatch: the manifest was modified after it was locked.\n"
            f"recorded  ={recorded}\nrecomputed={recomputed}\n"
            "A locked manifest is immutable; create a new protocol version instead."
        )
    return recorded


def verify_checkpoints(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Confirm every locked checkpoint exists and still matches its recorded digest."""
    models = manifest.get("models")
    if not isinstance(models, dict) or not models:
        raise ManifestError("Manifest defines no models to evaluate.")
    locked_seeds = manifest.get("seeds")
    if not isinstance(locked_seeds, list) or not locked_seeds:
        raise ManifestError("Manifest defines no locked seeds.")

    verified: list[dict[str, Any]] = []
    for name, entries in models.items():
        if not isinstance(entries, list) or not entries:
            raise ManifestError(f"Model {name!r} has no locked checkpoints.")
        seeds = [int(entry["seed"]) for entry in entries]
        if sorted(seeds) != sorted(int(seed) for seed in locked_seeds):
            raise ManifestError(
                f"Model {name!r} covers seeds {sorted(seeds)} but the manifest locks "
                f"{sorted(locked_seeds)}. Every locked seed must be evaluated; seeds must "
                "never be dropped or cherry-picked."
            )
        if len(set(seeds)) != len(seeds):
            raise ManifestError(f"Model {name!r} has duplicate seeds: {seeds}.")
        for entry in entries:
            checkpoint_path = Path(entry["checkpoint"])
            if not checkpoint_path.is_file():
                raise ManifestError(
                    f"Locked checkpoint is missing for {name}/seed_{entry['seed']}: {checkpoint_path}"
                )
            recorded = entry.get("checkpoint_sha256")
            if not recorded:
                raise ManifestError(
                    f"Manifest entry {name}/seed_{entry['seed']} has no checkpoint_sha256."
                )
            actual = file_sha256(checkpoint_path)
            if actual != recorded:
                raise ManifestError(
                    f"Checkpoint checksum mismatch for {name}/seed_{entry['seed']}:\n"
                    f"  path      = {checkpoint_path}\n"
                    f"  recorded  = {recorded}\n"
                    f"  actual    = {actual}\n"
                    "The checkpoint changed after Stage 12 locked it; refusing to evaluate."
                )
            verified.append(
                {
                    "experiment_name": name,
                    "seed": int(entry["seed"]),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": actual,
                }
            )
    return verified


def validate_manifest(path: str | Path) -> dict[str, Any]:
    """Full Stage-13 pre-flight: structure, hash, provenance, and checkpoint digests."""
    manifest = load_manifest(path)
    if int(manifest.get("stage", 0)) != 13:
        raise ManifestError(f"Manifest is not a Stage-13 manifest: stage={manifest.get('stage')!r}")
    if manifest.get("official_test_evaluated") is not False:
        raise ManifestError(
            "Manifest already records official_test_evaluated=True; the locked test "
            "evaluation has already been consumed for this manifest."
        )
    if not manifest.get("protocol_hash"):
        raise ManifestError("Manifest is missing the Stage-12 protocol_hash.")
    if not manifest.get("data_signature_hash"):
        raise ManifestError("Manifest is missing the data_signature_hash.")
    manifest_hash = verify_manifest_hash(manifest)
    checkpoints = verify_checkpoints(manifest)
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_hash,
        "protocol_hash": manifest["protocol_hash"],
        "seeds": sorted(int(seed) for seed in manifest["seeds"]),
        "checkpoints": checkpoints,
        "models": sorted(manifest["models"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the locked Stage-13 manifest.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    resolved = validate_manifest(args.manifest)
    print(f"Manifest SHA-256: {resolved['manifest_sha256']}")
    print(f"Protocol hash:    {resolved['protocol_hash']}")
    print(f"Models:           {resolved['models']}")
    print(f"Seeds:            {resolved['seeds']}")
    print(f"Checkpoints verified: {len(resolved['checkpoints'])}")
    print("STAGE 13 MANIFEST VALIDATION: PASS")
    print("Official test has NOT been loaded by this validator.")


if __name__ == "__main__":
    main()
