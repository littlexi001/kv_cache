#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any

import train_frontier_mode_router_v421_20260712 as frontier
import train_macro_mode_router_v418_20260712 as macro
import train_policy_action_planner_v378_20260712 as v378


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
FULL_ONLINE = 3.0988


def parse_float_list(spec: str) -> list[float]:
    return [float(item) for item in spec.split(",") if item.strip()]


def relabel_latency(
    rows: list[dict[str, Any]],
    candidate_by_key: dict[tuple[str, str, str], dict[str, Any]],
    actions: list[str],
    kv_penalty: float,
    latency_penalty: float,
    min_score_floor: float,
) -> list[dict[str, Any]]:
    out = copy.deepcopy(rows)
    for row in out:
        task = str(row["task"])
        sample_id = str(row["sample_id"])
        choices = []
        relaxed = []
        for action in actions:
            cand = candidate_by_key[(task, sample_id, action)]
            score = float(cand["score"])
            kv = float(cand["kv"])
            online = float(cand["online"])
            normalized_latency = online / max(1e-9, FULL_ONLINE)
            utility = score - kv_penalty * kv - latency_penalty * normalized_latency
            item = (utility, score, -kv, -online, action, cand)
            if score + 1e-12 >= min_score_floor * float(row["full_score"]):
                choices.append(item)
            relaxed.append(item)
        if not choices:
            choices = relaxed
        _, _, _, _, action, cand = max(choices)
        row["oracle_action"] = action
        row["oracle_score"] = float(cand["score"])
        row["oracle_kv"] = float(cand["kv"])
        row["oracle_online"] = float(cand["online"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--full-results", default="outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv")
    parser.add_argument("--output-dir", default="outputs/riskkv_v19_v424_latency_frontier_router_20260712")
    parser.add_argument("--config-out", default="configs/riskkv_task_policy_v424_latency_frontier_router_20260712.json")
    parser.add_argument("--base-action", default="frontier_050")
    parser.add_argument("--kv-limit", type=float, default=0.055)
    parser.add_argument("--quality-ratio", type=float, default=0.95)
    parser.add_argument("--speed-min", type=float, default=6.0)
    parser.add_argument("--min-score-floor", type=float, default=0.90)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--kv-penalties", default="0.35,0.75,1.5,3.0,6.0")
    parser.add_argument("--latency-penalties", default="0.0,0.25,0.5,1.0,2.0,4.0")
    args = parser.parse_args()

    root = Path(args.root)
    rows, candidate_by_key, specs = frontier.build_rows(root, args.full_results, args.base_action)
    actions = sorted(specs)
    names = macro.feature_names(rows)

    from sklearn.ensemble import RandomForestClassifier

    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for kv_penalty in parse_float_list(args.kv_penalties):
        for latency_penalty in parse_float_list(args.latency_penalties):
            labeled = relabel_latency(
                rows,
                candidate_by_key,
                actions,
                kv_penalty=kv_penalty,
                latency_penalty=latency_penalty,
                min_score_floor=args.min_score_floor,
            )
            train = [row for row in labeled if int(row["fold"]) not in {0, 1}]
            cal = [row for row in labeled if int(row["fold"]) == 1]
            model = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=424 + int(round(kv_penalty * 100)) + int(round(latency_penalty * 1000)),
                n_jobs=-1,
            )
            model.fit(macro.vectorize(train, names), [str(row["oracle_action"]) for row in train])
            for threshold in [round(i / 100, 2) for i in range(0, 101)] + [1.01]:
                pred_cal = macro.predict(cal, candidate_by_key, model, names, threshold)
                cal_summary = next(row for row in macro.summarize(pred_cal, "calibration") if row["task"] == "ALL")
                feasible = (
                    float(cal_summary["learned_vs_full"]) >= args.quality_ratio
                    and float(cal_summary["learned_kv"]) <= args.kv_limit
                    and float(cal_summary["learned_speed_vs_full"]) >= args.speed_min
                )
                trial = {
                    "kv_penalty": kv_penalty,
                    "latency_penalty": latency_penalty,
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
                    float(cal_summary["learned_score"]),
                    float(cal_summary["learned_speed_vs_full"]),
                    -float(cal_summary["learned_kv"]),
                    float(cal_summary["oracle_match_rate"]),
                )
                if best is None or key > best["key"]:
                    best = {
                        "key": key,
                        "kv_penalty": kv_penalty,
                        "latency_penalty": latency_penalty,
                        "threshold": threshold,
                        "model": model,
                        "labeled": labeled,
                    }

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
            row["kv_penalty"] = best["kv_penalty"]
            row["latency_penalty"] = best["latency_penalty"]
            row["threshold"] = best["threshold"]
        predictions.extend(pred)
        summaries.extend(macro.summarize(pred, split))

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    action_policy = v378.build_action_policy(root, specs)
    metadata = {
        "router_type": "latency_aware_frontier_router_v424",
        "feature_names": names,
        "confidence_fallback_threshold": best["threshold"],
        "kv_penalty": best["kv_penalty"],
        "latency_penalty": best["latency_penalty"],
        "base_action": args.base_action,
        "base_policy": Path(specs[args.base_action]["config"]).name,
        "available_actions": actions,
        "candidate_configs": {action: spec["config"] for action, spec in specs.items()},
        "uses_task_one_hot": False,
        "proxy_frontiers": True,
        "speed_min": args.speed_min,
    }
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"model": best["model"], "metadata": metadata}, handle)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "action_policy.json").write_text(json.dumps(action_policy, indent=2, ensure_ascii=False), encoding="utf-8")
    macro.write_csv(output_dir / "latency_penalty_trials.csv", trials)
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
            "v424: latency-aware frontier router. It uses the same frontier action space as v421, "
            "but labels actions with both KV and online-latency penalties."
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
