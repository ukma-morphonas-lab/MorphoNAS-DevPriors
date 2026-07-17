#!/usr/bin/env python3
"""Analyze a canonical paired-reward run with source-clustered uncertainty.

The point statistic is the existing Mantel-Haenszel odds ratio over exact
``(N,E)`` strata. Uncertainty is obtained by resampling source clusters and
carrying the grown row plus every nested random graph in both reward cells.
The same bootstrap draw therefore gives component intervals and a genuinely
paired interval for the ratio of odds ratios (RoR).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from MorphoNAS_DevPriors.reproducibility import (  # noqa: E402
    canonical_json_sha256,
    configure_torch_cpu,
    environment_fingerprint,
    set_early_runtime_environment,
    sha256_file,
    verify_locked_environment,
)

set_early_runtime_environment()

import numpy as np  # noqa: E402


Z95 = 1.959963984540054


def _sha256(path: str | Path) -> str:
    return sha256_file(path)


def _read_jsonl(path: Path, key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    rows = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = tuple(row[field] for field in key_fields)
            if key in rows:
                raise RuntimeError(f"duplicate key {key} in {path}:{line_number}")
            rows[key] = row
    return rows


def _write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _mh_from_cells(a, n_mn, c, n_rd) -> dict:
    keep = (n_mn > 0) & (n_rd > 0)
    a, n_mn, c, n_rd = a[keep], n_mn[keep], c[keep], n_rd[keep]
    b = n_mn - a
    d = n_rd - c
    total = n_mn + n_rd
    ri = a * d / total
    si = b * c / total
    numerator = float(np.sum(ri))
    denominator = float(np.sum(si))
    if numerator <= 0 or denominator <= 0:
        return {"or_mh": None, "note": "degenerate numerator or denominator"}

    point = numerator / denominator
    pi = (a + d) / total
    qi = (b + c) / total
    v_r = float(np.sum(pi * ri))
    v_rs = float(np.sum(pi * si + qi * ri))
    v_s = float(np.sum(qi * si))
    variance = (
        v_r / (2.0 * numerator * numerator)
        + v_rs / (2.0 * numerator * denominator)
        + v_s / (2.0 * denominator * denominator)
    )
    standard_se = math.sqrt(variance)
    return {
        "or_mh": point,
        "ln_or": math.log(point),
        "standard_rbg_se_ln": standard_se,
        "standard_rbg_ci": [
            point * math.exp(-Z95 * standard_se),
            point * math.exp(Z95 * standard_se),
        ],
        "n_strata_shared": int(np.sum(keep)),
    }


def _aggregate_mh(data: dict, weights: np.ndarray, side: int) -> float:
    shape_index = data["shape_index"]
    n_shapes = int(data["n_shapes"])
    grown = data["grown"][:, side]
    random_competent = data["random_competent"][:, side]
    random_total = data["random_total"]
    a = np.bincount(shape_index, weights=weights * grown, minlength=n_shapes)
    n_mn = np.bincount(shape_index, weights=weights, minlength=n_shapes)
    c = np.bincount(
        shape_index, weights=weights * random_competent, minlength=n_shapes
    )
    n_rd = np.bincount(shape_index, weights=weights * random_total, minlength=n_shapes)
    result = _mh_from_cells(a, n_mn, c, n_rd)
    return float(result["or_mh"]) if result.get("or_mh") is not None else math.nan


def _point_mh(data: dict, side: int) -> dict:
    weights = np.ones(data["n_sources"], dtype=float)
    shape_index = data["shape_index"]
    n_shapes = int(data["n_shapes"])
    a = np.bincount(
        shape_index, weights=weights * data["grown"][:, side], minlength=n_shapes
    )
    n_mn = np.bincount(shape_index, weights=weights, minlength=n_shapes)
    c = np.bincount(
        shape_index,
        weights=weights * data["random_competent"][:, side],
        minlength=n_shapes,
    )
    n_rd = np.bincount(
        shape_index, weights=weights * data["random_total"], minlength=n_shapes
    )
    return _mh_from_cells(a, n_mn, c, n_rd)


def _source_matched_point(data: dict, side: int) -> float | None:
    grown = data["grown"][:, side]
    competent = data["random_competent"][:, side]
    total = data["random_total"]
    denominator_n = 1.0 + total
    numerator = np.sum(grown * (total - competent) / denominator_n)
    denominator = np.sum((1.0 - grown) * competent / denominator_n)
    if numerator <= 0 or denominator <= 0:
        return None
    return float(numerator / denominator)


def _ci(values: np.ndarray) -> list[float | None]:
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return [None, None]
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def _valid_bootstrap_count(values: np.ndarray) -> int:
    return int(np.sum(np.isfinite(values) & (values > 0)))


def _require_sha256(value: object, context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{context} lacks a canonical SHA-256")


def _load_data(run_dir: Path, metadata: dict) -> tuple[dict, dict]:
    task_a, task_b = metadata["tasks"]
    source_rows = _read_jsonl(run_dir / "sources.jsonl", ("pair_source_id",))
    control_rows = _read_jsonl(
        run_dir / "controls.jsonl", ("pair_source_id", "k")
    )
    valid_sources = {
        key[0]: row for key, row in source_rows.items() if row.get("valid")
    }
    if len(source_rows) != int(metadata["requested_sources"]):
        raise RuntimeError(
            f"source row count {len(source_rows)} != requested "
            f"{metadata['requested_sources']}"
        )
    if int(metadata.get("trajectory_assertion_failures", 0)) != 0:
        raise RuntimeError("metadata reports trajectory assertion failures")
    source_ids = sorted(valid_sources)
    if not source_ids:
        raise RuntimeError("no valid grown sources")
    index = {source_id: position for position, source_id in enumerate(source_ids)}
    n = len(source_ids)
    shapes = [
        (
            int(valid_sources[source_id]["network_stats"]["neurons"]),
            int(valid_sources[source_id]["network_stats"]["connections"]),
        )
        for source_id in source_ids
    ]
    labels = {shape: label for label, shape in enumerate(sorted(set(shapes)))}
    shape_index = np.array([labels[shape] for shape in shapes], dtype=np.int32)
    grown = np.zeros((n, 2), dtype=float)
    grown_rewards = np.zeros((n, 2), dtype=float)
    random_competent = np.zeros((n, 2), dtype=float)
    random_total = np.zeros(n, dtype=float)
    random_rewards = np.full(
        (n, int(metadata["num_random_per_source"]), 2), np.nan, dtype=float
    )
    random_valid = np.zeros(
        (n, int(metadata["num_random_per_source"])), dtype=bool
    )
    source_hashes = []
    topology_shape_matches = 0
    topology_shape_mismatches = 0
    for source_id in source_ids:
        position = index[source_id]
        row = valid_sources[source_id]
        if row.get("topology_shape_match"):
            topology_shape_matches += 1
        else:
            topology_shape_mismatches += 1
        if set(row.get("scores", {})) != {task_a, task_b}:
            raise RuntimeError(f"source {source_id} has incomplete task scores")
        if len(row.get("eval_seeds", [])) != int(metadata["rollouts"]):
            raise RuntimeError(f"source {source_id} has the wrong rollout-seed count")
        grown[position, 0] = int(bool(row["scores"][task_a]["competent"]))
        grown[position, 1] = int(bool(row["scores"][task_b]["competent"]))
        grown_rewards[position, 0] = float(row["scores"][task_a]["mean_reward"])
        grown_rewards[position, 1] = float(row["scores"][task_b]["mean_reward"])
        if not row.get("trajectory_equality_verified"):
            raise RuntimeError(f"source {source_id} lacks trajectory equality verification")
        _require_sha256(row.get("graph_sha256"), f"source {source_id} graph")
        _require_sha256(row.get("genome_sha256"), f"source {source_id} genome")
        _require_sha256(row.get("anchor_sha256"), f"source {source_id} anchor")
        _require_sha256(row.get("trajectory_sha256"), f"source {source_id} trajectory")
        source_hashes.append(row["trajectory_sha256"])

    valid_controls = 0
    invalid_controls = 0
    control_hashes = []
    requested_keys = {
        (source_id, k)
        for source_id in source_ids
        for k in range(int(metadata["num_random_per_source"]))
    }
    actual_keys = set(control_rows)
    if actual_keys != requested_keys:
        missing = sorted(requested_keys - actual_keys)[:10]
        extra = sorted(actual_keys - requested_keys)[:10]
        raise RuntimeError(
            f"control request set is incomplete or incompatible; "
            f"missing(first 10)={missing}, extra(first 10)={extra}"
        )
    for (source_id, k), row in control_rows.items():
        if source_id not in index:
            raise RuntimeError(f"control references unknown/invalid source {source_id}")
        expected_seed = int(metadata["control_base_seed"]) + source_id * 100 + k
        if int(row["random_seed"]) != expected_seed:
            raise RuntimeError(
                f"control seed mismatch for {source_id}/{k}: "
                f"{row['random_seed']} != {expected_seed}"
            )
        if not row.get("valid"):
            invalid_controls += 1
            continue
        if not row.get("trajectory_equality_verified"):
            raise RuntimeError(f"control {source_id}/{row['k']} lacks trajectory verification")
        if set(row.get("scores", {})) != {task_a, task_b}:
            raise RuntimeError(f"control {source_id}/{k} has incomplete task scores")
        if len(row.get("eval_seeds", [])) != int(metadata["rollouts"]):
            raise RuntimeError(f"control {source_id}/{k} has the wrong rollout-seed count")
        if row["eval_seeds"] != valid_sources[source_id]["eval_seeds"]:
            raise RuntimeError(
                f"control {source_id}/{k} does not reuse its grown source's seeds"
            )
        position = index[source_id]
        shape = (
            int(row["network_stats"]["neurons"]),
            int(row["network_stats"]["connections"]),
        )
        if shape != shapes[position]:
            raise RuntimeError(f"control shape mismatch for source {source_id}: {shape}")
        random_total[position] += 1
        random_competent[position, 0] += int(bool(row["scores"][task_a]["competent"]))
        random_competent[position, 1] += int(bool(row["scores"][task_b]["competent"]))
        random_rewards[position, k, 0] = float(row["scores"][task_a]["mean_reward"])
        random_rewards[position, k, 1] = float(row["scores"][task_b]["mean_reward"])
        random_valid[position, k] = True
        _require_sha256(row.get("graph_sha256"), f"control {source_id}/{k} graph")
        _require_sha256(
            row.get("trajectory_sha256"), f"control {source_id}/{k} trajectory"
        )
        control_hashes.append(row["trajectory_sha256"])
        valid_controls += 1

    data = {
        "n_sources": n,
        "shape_index": shape_index,
        "n_shapes": len(labels),
        "grown": grown,
        "grown_rewards": grown_rewards,
        "random_competent": random_competent,
        "random_total": random_total,
        "random_rewards": random_rewards,
        "random_valid": random_valid,
    }
    audit = {
        "source_rows": len(source_rows),
        "valid_sources": n,
        "invalid_sources": len(source_rows) - n,
        "valid_controls": valid_controls,
        "invalid_controls": invalid_controls,
        "sources_with_valid_controls": int(np.sum(random_total > 0)),
        "min_valid_controls_per_source": int(np.min(random_total)),
        "max_valid_controls_per_source": int(np.max(random_total)),
        "requested_controls": len(control_rows),
        "requested_controls_per_source": int(metadata["num_random_per_source"]),
        "lockstep_source_trajectories": len(source_hashes),
        "lockstep_control_trajectories": len(control_hashes),
        "unique_source_trajectory_hashes": len(set(source_hashes)),
        "unique_control_trajectory_hashes": len(set(control_hashes)),
        "trajectory_assertion_failures": metadata.get("trajectory_assertion_failures", 0),
        "historical_topology_shape_matches": topology_shape_matches,
        "historical_topology_shape_mismatches": topology_shape_mismatches,
        "historical_topology_role": "audit only; controls use freshly regrown (N,E)",
    }
    return data, audit


def _data_at_threshold(data: dict, side: int, cutoff: float) -> dict:
    random_valid = data["random_valid"]
    competent = (data["random_rewards"][:, :, side] >= cutoff) & random_valid
    return {
        "n_sources": data["n_sources"],
        "shape_index": data["shape_index"],
        "n_shapes": data["n_shapes"],
        "grown": (data["grown_rewards"][:, side] >= cutoff).astype(float)[:, None],
        "random_competent": np.sum(competent, axis=1, dtype=float)[:, None],
        "random_total": np.sum(random_valid, axis=1, dtype=float),
    }


def _quantile_cutoff(rewards: np.ndarray, target_rate: float) -> float:
    ordered = np.sort(np.asarray(rewards, dtype=float))[::-1]
    count = max(1, min(len(ordered), round(target_rate * len(ordered))))
    return float(ordered[count - 1])


def analyze(run_dir: Path, bootstrap_replicates: int, bootstrap_seed: int) -> dict:
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise RuntimeError(f"run is not complete: status={metadata.get('status')}")
    for filename in ("sources.jsonl", "controls.jsonl"):
        expected = metadata.get("science_file_sha256", {}).get(filename)
        actual = _sha256(run_dir / filename)
        if expected != actual:
            raise RuntimeError(
                f"{filename} hash {actual} != completed-run metadata {expected}"
            )
    task_a, task_b = metadata["tasks"]
    data, pairing_audit = _load_data(run_dir, metadata)
    n = int(data["n_sources"])

    point_a = _point_mh(data, 0)
    point_b = _point_mh(data, 1)
    source_point_a = _source_matched_point(data, 0)
    source_point_b = _source_matched_point(data, 1)

    sweep_panels: list[dict] = []
    if metadata["pair"] == "acrobot":
        seen: set[float] = set()
        for target_rate in (0.02, 0.03, 0.05, 0.08, 0.12):
            cutoff = _quantile_cutoff(data["grown_rewards"][:, 0], target_rate)
            if cutoff in seen:
                continue
            seen.add(cutoff)
            threshold_data = _data_at_threshold(data, 0, cutoff)
            sweep_panels.append(
                {
                    "target_grown_rate": target_rate,
                    "cutoff": cutoff,
                    "data": threshold_data,
                    "point": _point_mh(threshold_data, 0),
                    "source_point": _source_matched_point(threshold_data, 0),
                    "bootstrap": np.empty(bootstrap_replicates, dtype=float),
                }
            )

    rng = np.random.default_rng(bootstrap_seed)
    boot_a = np.empty(bootstrap_replicates, dtype=float)
    boot_b = np.empty(bootstrap_replicates, dtype=float)
    for replicate in range(bootstrap_replicates):
        sampled = rng.integers(0, n, size=n)
        multiplicity = np.bincount(sampled, minlength=n).astype(float)
        boot_a[replicate] = _aggregate_mh(data, multiplicity, 0)
        boot_b[replicate] = _aggregate_mh(data, multiplicity, 1)
        for panel in sweep_panels:
            panel["bootstrap"][replicate] = _aggregate_mh(
                panel["data"], multiplicity, 0
            )
    boot_ror = boot_a / boot_b

    grown = data["grown"]
    random_competent = data["random_competent"]
    random_total = data["random_total"]

    def threshold_for(task: str) -> float:
        for bound in metadata["task_specs"][task]["bounds"]:
            if bound["name"] == "weak":
                return bound["high"]
        raise RuntimeError(f"task {task} has no weak stratum")

    def component(
        task: str,
        side: int,
        point: dict,
        boot: np.ndarray,
        source_point: float | None,
    ) -> dict:
        return {
            "task": task,
            "threshold": threshold_for(task),
            "threshold_provenance": metadata["task_specs"][task]["note"],
            "grown": {
                "competent": int(np.sum(grown[:, side])),
                "total": n,
            },
            "random": {
                "competent": int(np.sum(random_competent[:, side])),
                "total_valid": int(np.sum(random_total)),
                "nominal_requested": n * int(metadata["num_random_per_source"]),
            },
            "ne_stratified_mh": {
                **point,
                "source_cluster_percentile_ci": _ci(boot),
                "bootstrap_valid_replicates": _valid_bootstrap_count(boot),
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": bootstrap_seed,
            },
            "source_stratified_mh_sensitivity": source_point,
        }

    paired_point = None
    paired_note = None
    if point_a.get("or_mh") is not None and point_b.get("or_mh") is not None:
        paired_point = float(point_a["or_mh"] / point_b["or_mh"])
    else:
        paired_note = "one or both component MH estimators are degenerate"

    threshold_sweep = None
    if sweep_panels:
        rows = []
        for panel in sweep_panels:
            panel_data = panel["data"]
            point = panel["point"]
            interval = _ci(panel["bootstrap"])
            rows.append(
                {
                    "target_grown_rate": panel["target_grown_rate"],
                    "cutoff": panel["cutoff"],
                    "grown": {
                        "competent": int(np.sum(panel_data["grown"][:, 0])),
                        "total": n,
                    },
                    "random": {
                        "competent": int(
                            np.sum(panel_data["random_competent"][:, 0])
                        ),
                        "total_valid": int(np.sum(panel_data["random_total"])),
                    },
                    "ne_stratified_mh": {
                        **point,
                        "source_cluster_percentile_ci": interval,
                        "bootstrap_valid_replicates": _valid_bootstrap_count(
                            panel["bootstrap"]
                        ),
                        "bootstrap_replicates": bootstrap_replicates,
                        "bootstrap_seed": bootstrap_seed,
                    },
                    "source_stratified_mh_sensitivity": panel["source_point"],
                    "source_cluster_ci_above_one": (
                        interval[0] is not None and interval[0] > 1.0
                    ),
                }
            )
        threshold_sweep = {
            "task": task_a,
            "selection": "cutoffs targeting grown pass rates fixed at 2%, 3%, 5%, 8%, and 12% within this rerun",
            "rows": rows,
            "all_source_cluster_cis_above_one": all(
                row["source_cluster_ci_above_one"] for row in rows
            ),
        }

    result = {
        "schema_version": 2,
        "experiment": "paired_reward_analysis",
        "pair": metadata["pair"],
        "tasks": [task_a, task_b],
        "analysis_contract": {
            "point_estimator": "Mantel-Haenszel odds ratio over exact (N,E) strata",
            "primary_uncertainty": "nonparametric source-cluster percentile bootstrap",
            "cluster_contents": "one grown row and all nested random-control requests; invalid requests remain source-linked through the observed valid-control count",
            "paired_contrast": "same source-cluster bootstrap draw for both cells",
        },
        "components": {
            task_a: component(task_a, 0, point_a, boot_a, source_point_a),
            task_b: component(task_b, 1, point_b, boot_b, source_point_b),
        },
        "paired_ratio_of_odds_ratios": {
            "point": paired_point,
            "source_cluster_percentile_ci": _ci(boot_ror),
            "bootstrap_valid_replicates": _valid_bootstrap_count(boot_ror),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "descriptive_only": False,
            "note": paired_note,
        },
        "native_acrobot_threshold_sweep": threshold_sweep,
        "pairing_audit": pairing_audit,
        "input_files": {
            "sources": {"path": "sources.jsonl", "sha256": _sha256(run_dir / "sources.jsonl")},
            "controls": {"path": "controls.jsonl", "sha256": _sha256(run_dir / "controls.jsonl")},
        },
        "immutable_run_config_sha256": canonical_json_sha256(
            metadata["immutable_config"]
        ),
        "analyzer_sha256": _sha256(__file__),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_717)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--environment-lock",
        default=str(REPO_ROOT / "reproduction/environment.json"),
    )
    parser.add_argument("--allow-unlocked-environment", action="store_true")
    args = parser.parse_args()
    if args.bootstrap <= 0:
        raise SystemExit("--bootstrap must be positive")
    if args.allow_unlocked_environment:
        configure_torch_cpu()
        scientific_environment = environment_fingerprint()
        scientific_environment.update(
            {
                "verified": False,
                "canonical": False,
                "warning": "unlocked development analysis; not publishable evidence",
            }
        )
    else:
        scientific_environment = verify_locked_environment(args.environment_lock)
        scientific_environment["canonical"] = True
    run_dir = Path(args.run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if not args.allow_unlocked_environment and not metadata.get(
        "canonical_environment"
    ):
        raise SystemExit("refusing canonical analysis of an unlocked run")
    result = analyze(run_dir, args.bootstrap, args.seed)
    output = Path(args.out) if args.out else run_dir / "paired_reward_results.json"
    _write_json(output, result)
    provenance = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "hostname": socket.gethostname(),
        "analysis_environment": scientific_environment,
        "analyzer_sha256": _sha256(__file__),
        "metadata_sha256": _sha256(run_dir / "metadata.json"),
        "results_sha256": _sha256(output),
    }
    _write_json(run_dir / "analysis_provenance.private.json", provenance)
    public_provenance = dict(provenance)
    public_provenance.pop("hostname")
    _write_json(run_dir / "analysis_provenance.public.json", public_provenance)

    task_a, task_b = result["tasks"]
    for task in (task_a, task_b):
        component = result["components"][task]
        point = component["ne_stratified_mh"]["or_mh"]
        lo, hi = component["ne_stratified_mh"]["source_cluster_percentile_ci"]
        if point is None:
            interval = "degenerate"
        elif lo is None or hi is None:
            interval = f"MH OR {point:.6g} [bootstrap degenerate]"
        else:
            interval = f"MH OR {point:.6g} [{lo:.6g}, {hi:.6g}]"
        print(
            f"{task}: {component['grown']['competent']}/{component['grown']['total']} "
            f"vs {component['random']['competent']}/{component['random']['total_valid']}; "
            f"{interval}"
        )
    ror = result["paired_ratio_of_odds_ratios"]
    lo, hi = ror["source_cluster_percentile_ci"]
    if ror["point"] is None:
        print("paired RoR degenerate")
    elif lo is None or hi is None:
        print(f"paired RoR {ror['point']:.6g} [bootstrap degenerate]")
    else:
        print(f"paired RoR {ror['point']:.6g} [{lo:.6g}, {hi:.6g}]")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
