#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import train_macro_mode_router_v418_20260712 as macro
import train_policy_action_planner_v378_20260712 as v378


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
FULL_SCORE = 0.36581658127460975
FULL_ONLINE = 3.0988

RESULT_BY_CONFIG = {
    "riskkv_task_policy_v368_direct_operator_extreme_mix_20260712.json": "outputs/riskkv_v19_v368_direct_operator_extreme_mix_all_confirm_m100_20260712_lowkv_confirm_m100_bDyn_pDyn/task_results.csv",
    "riskkv_task_policy_v381_policy_multiclass_nopost_20260712.json": "outputs/riskkv_v19_v381_policy_multiclass_nopost_20260712_policy_multiclass_nopost_v381_m100_bDyn_pDyn/task_results.csv",
    "riskkv_task_policy_v389_m100_task_knapsack_v2_20260712.json": "outputs/riskkv_v19_v389_m100_task_knapsack_v2_20260712_m100_task_knapsack_v2_v389_m100_bDyn_pDyn/task_results.csv",
    "riskkv_task_policy_v393_m100_task_knapsack_v385_20260712.json": "outputs/riskkv_v19_v393_m100_task_knapsack_v385_20260712_m100_task_knapsack_v385_v393_m100_bDyn_pDyn/task_results.csv",
    "riskkv_task_policy_v394_m100_task_knapsack10_exact_20260712.json": "outputs/riskkv_v19_v394_m100_task_knapsack10_exact_20260712_m100_task_knapsack10_exact_v394_m100_bDyn_pDyn/task_results.csv",
    "riskkv_task_policy_v395_m100_task_knapsack075_exact_20260712.json": "outputs/riskkv_v19_v395_m100_task_knapsack075_exact_20260712_m100_task_knapsack075_exact_v395_m100_bDyn_pDyn/task_results.csv",
    "riskkv_task_policy_v396_m100_task_knapsack05_exact_20260712.json": "outputs/riskkv_v19_v396_m100_task_knapsack05_exact_20260712_m100_task_knapsack05_exact_v396_m100_bDyn_pDyn/task_results.csv",
    "riskkv_task_policy_v413_legacy_b128_20260712.json": "outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/riskkv_question_aware_b128/task_results.csv",
    "riskkv_task_policy_v413_legacy_b256_20260712.json": "outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/riskkv_question_aware_b256/task_results.csv",
    "riskkv_task_policy_v413_legacy_b512_20260712.json": "outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/riskkv_question_aware_b512/task_results.csv",
}

FRONTIER_CANDIDATES = {
    "operator_direct": "configs/riskkv_task_policy_v368_direct_operator_extreme_mix_20260712.json",
    "frontier_030": "configs/riskkv_task_policy_v417_expanded_knapsack030_20260712.json",
    "frontier_035": "configs/riskkv_task_policy_v413_expanded_knapsack035_20260712.json",
    "frontier_040": "configs/riskkv_task_policy_v414_expanded_knapsack040_20260712.json",
    "frontier_045": "configs/riskkv_task_policy_v415_expanded_knapsack045_20260712.json",
    "frontier_050": "configs/riskkv_task_policy_v396_m100_task_knapsack05_exact_20260712.json",
    "frontier_075": "configs/riskkv_task_policy_v395_m100_task_knapsack075_exact_20260712.json",
    "frontier_100": "configs/riskkv_task_policy_v394_m100_task_knapsack10_exact_20260712.json",
    "legacy_b128": "configs/riskkv_task_policy_v413_legacy_b128_20260712.json",
    "legacy_b256": "configs/riskkv_task_policy_v413_legacy_b256_20260712.json",
    "legacy_b512": "configs/riskkv_task_policy_v413_legacy_b512_20260712.json",
}


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def read_table(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["task"], row["sample_id"]): row for row in csv.DictReader(handle)}


def policy_path(root: Path, spec: str) -> Path:
    path = Path(spec)
    return path if path.is_absolute() else root / path


@lru_cache(maxsize=2048)
def load_json_cached(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_result_source(root: Path, config_path: Path, target_task: str) -> tuple[str, str]:
    name = config_path.name
    if name in RESULT_BY_CONFIG:
        return name, target_task
    payload = load_json_cached(str(config_path))
    if isinstance(payload, dict) and "__extends" in payload:
        parent = Path(str(payload["__extends"]))
        if not parent.is_absolute():
            parent = config_path.parent / parent
        source_config, source_task = resolve_result_source(root, parent, target_task)
    elif name in RESULT_BY_CONFIG:
        source_config, source_task = name, target_task
    else:
        source_config, source_task = name, target_task

    if isinstance(payload, dict):
        task_sources = payload.get("__task_sources", {})
        if isinstance(task_sources, dict) and target_task in task_sources:
            entry = task_sources[target_task]
            if isinstance(entry, str):
                child_spec = entry
                child_task = target_task
            elif isinstance(entry, dict):
                child_spec = str(entry.get("policy", ""))
                child_task = str(entry.get("task", target_task))
            else:
                raise ValueError(f"Unsupported task source in {config_path}: {entry!r}")
            child_path = Path(child_spec)
            if not child_path.is_absolute():
                child_path = config_path.parent / child_path
            source_config, source_task = resolve_result_source(root, child_path, target_task if child_task == "*" else child_task)
            if child_task == "*":
                source_task = target_task
    return source_config, source_task


def synthesize_action_rows(root: Path, config_rel: str, tables: dict[str, dict[tuple[str, str], dict[str, str]]]) -> dict[tuple[str, str], dict[str, str]]:
    config_path = policy_path(root, config_rel)
    direct_rel = RESULT_BY_CONFIG.get(config_path.name)
    if direct_rel and (root / direct_rel).exists():
        if config_path.name not in tables:
            tables[config_path.name] = read_table(root / direct_rel)
        return tables[config_path.name]

    out: dict[tuple[str, str], dict[str, str]] = {}
    full = read_table(root / "outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv")
    for task, sample_id in sorted(full):
        source_config, source_task = resolve_result_source(root, config_path, task)
        rel = RESULT_BY_CONFIG.get(source_config)
        if not rel:
            raise FileNotFoundError(f"No result table registered for {source_config} while resolving {config_rel}/{task}")
        if source_config not in tables:
            tables[source_config] = read_table(root / rel)
        table = tables[source_config]
        row = table.get((source_task, sample_id))
        if row is None:
            row = table.get((task, sample_id))
        if row is None:
            continue
        copied = dict(row)
        copied["task"] = task
        copied["sample_id"] = sample_id
        copied["proxy_source_config"] = source_config
        copied["proxy_source_task"] = source_task
        out[(task, sample_id)] = copied
    return out


def build_rows(root: Path, full_results: str, base_action: str) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]], dict[str, dict[str, str]]]:
    full = read_table(root / full_results)
    cache: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    action_tables = {
        action: synthesize_action_rows(root, config, cache)
        for action, config in FRONTIER_CANDIDATES.items()
        if policy_path(root, config).exists()
    }
    if base_action not in action_tables:
        raise FileNotFoundError(f"base action {base_action!r} is unavailable")
    common = set(full) & set(action_tables[base_action])
    for table in action_tables.values():
        common &= set(table)

    rows: list[dict[str, Any]] = []
    candidate_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for task, sample_id in sorted(common):
        base_row = action_tables[base_action][(task, sample_id)]
        full_row = full[(task, sample_id)]
        row = {
            "task": task,
            "sample_id": sample_id,
            "fold": v378.fold_for_key(task, sample_id),
            "full_score": fnum(full_row, "score"),
            "base_action": base_action,
            "base_score": fnum(base_row, "score"),
            "base_kv": fnum(base_row, "keep_fraction"),
            "base_online": fnum(base_row, "online_seconds"),
        }
        for feature in v378.RUNTIME_NUMERIC_FEATURES:
            row[feature] = fnum(base_row, feature)
        rows.append(row)
        for action, table in action_tables.items():
            cand = table[(task, sample_id)]
            candidate_by_key[(task, sample_id, action)] = {
                "action": action,
                "score": fnum(cand, "score"),
                "kv": fnum(cand, "keep_fraction"),
                "online": fnum(cand, "online_seconds"),
            }
    specs = {
        action: {"config": config, "results": RESULT_BY_CONFIG.get(Path(config).name, f"proxy:{config}")}
        for action, config in FRONTIER_CANDIDATES.items()
        if action in action_tables
    }
    return rows, candidate_by_key, specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--full-results", default="outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv")
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_v421_frontier_mode_router_20260712")
    parser.add_argument("--config-out", default="configs/riskkv_task_policy_v421_frontier_mode_router_20260712.json")
    parser.add_argument("--base-action", default="frontier_050")
    parser.add_argument("--kv-limit", type=float, default=0.035)
    parser.add_argument("--quality-ratio", type=float, default=0.95)
    parser.add_argument("--min-score-floor", type=float, default=0.90)
    parser.add_argument("--n-estimators", type=int, default=240)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--penalties", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,8.0")
    args = parser.parse_args()

    root = Path(args.root)
    rows, candidate_by_key, specs = build_rows(root, args.full_results, args.base_action)
    actions = sorted(specs)
    names = macro.feature_names(rows)

    from sklearn.ensemble import RandomForestClassifier

    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for penalty in macro.parse_float_list(args.penalties):
        labeled = macro.relabel(rows, candidate_by_key, actions, penalty, args.min_score_floor)
        train = [row for row in labeled if int(row["fold"]) not in {0, 1}]
        cal = [row for row in labeled if int(row["fold"]) == 1]
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=421 + int(round(penalty * 100)),
            n_jobs=-1,
        )
        model.fit(macro.vectorize(train, names), [str(row["oracle_action"]) for row in train])
        for threshold in [round(i / 100, 2) for i in range(0, 101)] + [1.01]:
            pred_cal = macro.predict(cal, candidate_by_key, model, names, threshold)
            cal_summary = next(row for row in macro.summarize(pred_cal, "calibration") if row["task"] == "ALL")
            feasible = (
                float(cal_summary["learned_vs_full"]) >= args.quality_ratio
                and float(cal_summary["learned_kv"]) <= args.kv_limit
            )
            trial = {
                "penalty": penalty,
                "threshold": threshold,
                "cal_score": cal_summary["learned_score"],
                "cal_vs_full": cal_summary["learned_vs_full"],
                "cal_kv": cal_summary["learned_kv"],
                "cal_speed": cal_summary["learned_speed_vs_full"],
                "cal_fallback": cal_summary["fallback_rate"],
                "feasible": int(feasible),
            }
            trials.append(trial)
            key = (
                int(feasible),
                -float(cal_summary["learned_kv"]),
                float(cal_summary["learned_score"]),
                float(cal_summary["oracle_match_rate"]),
            )
            if best is None or key > best["key"]:
                best = {"key": key, "penalty": penalty, "threshold": threshold, "model": model, "labeled": labeled}

    assert best is not None
    predictions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for split, split_rows in {
        "train": [row for row in best["labeled"] if int(row["fold"]) not in {0, 1}],
        "calibration": [row for row in best["labeled"] if int(row["fold"]) == 1],
        "test": [row for row in best["labeled"] if int(row["fold"]) == 0],
        "all": best["labeled"],
    }.items():
        pred = macro.predict(split_rows, candidate_by_key, best["model"], names, float(best["threshold"]))
        for row in pred:
            row["split"] = split
            row["penalty"] = best["penalty"]
            row["threshold"] = best["threshold"]
        predictions.extend(pred)
        summaries.extend(macro.summarize(pred, split))

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    action_policy = v378.build_action_policy(root, specs)
    metadata = {
        "router_type": "frontier_mode_router_v421",
        "feature_names": names,
        "confidence_fallback_threshold": best["threshold"],
        "cost_penalty": best["penalty"],
        "base_action": args.base_action,
        "base_policy": Path(specs[args.base_action]["config"]).name,
        "available_actions": actions,
        "candidate_configs": {action: spec["config"] for action, spec in specs.items()},
        "uses_task_one_hot": False,
        "proxy_frontiers": True,
    }
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"model": best["model"], "metadata": metadata}, handle)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "action_policy.json").write_text(json.dumps(action_policy, indent=2, ensure_ascii=False), encoding="utf-8")
    macro.write_csv(output_dir / "penalty_threshold_trials.csv", trials)
    macro.write_csv(output_dir / "router_predictions.csv", predictions)
    macro.write_csv(output_dir / "router_summary.csv", summaries)
    macro.write_csv(
        output_dir / "feature_importance.csv",
        [
            {"feature": name, "importance": float(value)}
            for name, value in sorted(zip(names, best["model"].feature_importances_), key=lambda item: float(item[1]), reverse=True)
        ],
    )
    config = {
        "__extends": Path(specs[args.base_action]["config"]).name,
        "__comment": (
            "v421: frontier-mode router. It treats 3.0/3.5/4.0/4.5/5.0/7.5/10% task frontiers "
            "and legacy sparse retrieval as actions, then selects the lowest predicted-safe mode from runtime features."
        ),
        "__overlay_all_tasks": {
            "action_router": True,
            "ours_action_router_mode": "learned_budget_overlay_v1",
            "ours_learned_router_model_path": str((output_dir / "model.pkl").relative_to(root)),
            "ours_learned_router_action_policy_json": str((output_dir / "action_policy.json").relative_to(root)),
            "ours_learned_router_confidence_threshold": best["threshold"],
            "ours_learned_router_default_action": "reference",
            "ours_learned_router_base_action_router_mode": "off",
        },
    }
    config_path = root / args.config_out
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    all_summary = next(row for row in summaries if row["split"] == "all" and row["task"] == "ALL")
    cal_summary = next(row for row in summaries if row["split"] == "calibration" and row["task"] == "ALL")
    test_summary = next(row for row in summaries if row["split"] == "test" and row["task"] == "ALL")
    print(output_dir)
    print(config_path)
    print(json.dumps({"all": all_summary, "calibration": cal_summary, "test": test_summary, "metadata": metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
