#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MANIFEST = SCRIPT_DIR / "icml_runtime_manifest_20260707.json"
SUMMARIZER = SCRIPT_DIR / "summarize_runtime_scaling_20260707.py"


def load_summarizer() -> object:
    spec = importlib.util.spec_from_file_location("summarize_runtime_scaling_20260707", SUMMARIZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SUMMARIZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    summarizer = load_summarizer()

    for rel_path in manifest["local_files_to_sync"]:
        require((PROJECT_ROOT / rel_path).exists(), f"missing sync file: {rel_path}", errors)
    for rel_path in manifest["local_recovery_scripts"]:
        require((PROJECT_ROOT / rel_path).exists(), f"missing recovery script: {rel_path}", errors)

    methods = set(summarizer.METHODS)
    require("variable_budget_kv_planner" in methods, "summarizer missing variable_budget_kv_planner", errors)
    require("output_level_risk_kv_planner" in methods, "summarizer missing output_level_risk_kv_planner", errors)

    summary_paths = {
        path
        for paths in summarizer.EXPERIMENTS.values()
        for path in paths
    }
    for experiment in manifest["primary_experiments"]:
        result_path = f"{experiment['output_dir']}/results.csv"
        require(result_path in summary_paths, f"summarizer missing primary output: {result_path}", errors)
        require(experiment["method"] in methods, f"summarizer missing method for {experiment['name']}: {experiment['method']}", errors)

    for experiment in manifest["pending_scaling_experiments"]:
        if "output_dir" not in experiment:
            continue
        result_path = f"{experiment['output_dir']}/results.csv"
        require(result_path in summary_paths, f"summarizer missing scaling output: {result_path}", errors)
        require(experiment["method"] in methods, f"summarizer missing method for {experiment['name']}: {experiment['method']}", errors)

    launch_script = PROJECT_ROOT / "scripts/run_variable_budget_runtime_sweep_20260707.sh"
    launch_text = launch_script.read_text(encoding="utf-8")
    require("--runtime_methods full_kv_cache,variable_budget_kv_planner" in launch_text, "variable sweep is not method-filtered", errors)
    require("--variable_budget_tail_threshold \"$threshold\"" in launch_text, "variable sweep does not parameterize threshold", errors)
    require("conformal_auto" in launch_text, "variable sweep missing conformal_auto runs", errors)
    expansion_script = PROJECT_ROOT / "scripts/run_variable_budget_longbench_mixed_expansion_20260707.sh"
    require(expansion_script.exists(), "missing LongBench/Mixed expansion script", errors)
    expansion_text = expansion_script.read_text(encoding="utf-8")
    require("--max_examples_per_task 8" in expansion_text, "expansion script missing LongBench m8", errors)
    require("--max_examples_per_task 2" in expansion_text, "expansion script missing Mixed13 m2", errors)
    require("conformal_auto" in expansion_text, "expansion script missing conformal_auto runs", errors)
    variable_scaling_script = PROJECT_ROOT / "scripts/run_variable_budget_ruler_scaling_20260707.sh"
    require(variable_scaling_script.exists(), "missing variable-budget RULER scaling script", errors)
    variable_scaling_text = variable_scaling_script.read_text(encoding="utf-8")
    require(
        "--runtime_methods full_kv_cache,variable_budget_kv_planner" in variable_scaling_text,
        "variable RULER scaling is not method-filtered",
        errors,
    )
    require("case_start" in variable_scaling_text and "case_limit" in variable_scaling_text, "variable RULER scaling missing 16k sharding", errors)
    variable_expansion_script = PROJECT_ROOT / "scripts/run_variable_budget_ruler_expansion_m5_m3_20260707.sh"
    require(variable_expansion_script.exists(), "missing variable-budget RULER m5/m3 expansion script", errors)
    variable_expansion_text = variable_expansion_script.read_text(encoding="utf-8")
    require(
        "--runtime_methods full_kv_cache,variable_budget_kv_planner" in variable_expansion_text,
        "variable RULER m5/m3 expansion is not method-filtered",
        errors,
    )
    require("run_ruler 0 4096 5" in variable_expansion_text, "variable RULER m5/m3 expansion missing 4k m5 runs", errors)
    require("run_ruler 5 8192 5" in variable_expansion_text, "variable RULER m5/m3 expansion missing 8k m5 runs", errors)
    require("case2" in variable_expansion_text or " 2 " in variable_expansion_text, "variable RULER m5/m3 expansion missing 16k case2", errors)
    minsafe_m5_script = PROJECT_ROOT / "scripts/run_variable_budget_ruler_m5_minsafe_20260707.sh"
    require(minsafe_m5_script.exists(), "missing variable-budget RULER m5 min-safe script", errors)
    minsafe_m5_text = minsafe_m5_script.read_text(encoding="utf-8")
    require(
        "--runtime_methods full_kv_cache,variable_budget_kv_planner" in minsafe_m5_text,
        "variable RULER m5 min-safe script is not method-filtered",
        errors,
    )
    require("MINSAFE" in minsafe_m5_text, "variable RULER m5 min-safe script missing min-safe planner", errors)
    floor2_m5_script = PROJECT_ROOT / "scripts/run_variable_budget_ruler_m5_conformal_floor2_20260707.sh"
    require(floor2_m5_script.exists(), "missing variable-budget RULER m5 conformal floor2 script", errors)
    floor2_m5_text = floor2_m5_script.read_text(encoding="utf-8")
    require(
        "--runtime_methods full_kv_cache,variable_budget_kv_planner" in floor2_m5_text,
        "variable RULER m5 conformal floor2 script is not method-filtered",
        errors,
    )
    require("--variable_budget_min_budget 2" in floor2_m5_text, "variable RULER m5 conformal floor2 script missing k2 floor", errors)
    recovery_script = PROJECT_ROOT / "scripts/run_variable_budget_16k_missing_recovery_20260707.sh"
    require(recovery_script.exists(), "missing variable-budget 16k recovery script", errors)
    recovery_text = recovery_script.read_text(encoding="utf-8")
    require("RECOVERY_GPUS" in recovery_text, "16k recovery script does not allow GPU selection", errors)
    require("--runtime_methods full_kv_cache,variable_budget_kv_planner" in recovery_text, "16k recovery is not method-filtered", errors)
    for experiment_name in [
        "qwen8b_longbench_m8_conformal_auto",
        "qwen8b_mixed13_m2_conformal_auto",
        "qwen8b_ruler4k_m3_var_bestcal_tail035",
        "qwen8b_ruler8k_m3_var_bestcal_tail035",
        "qwen8b_ruler16k_m2_var_bestcal_tail035_sharded",
        "qwen8b_ruler16k_m2_conformal_auto_sharded",
        "qwen8b_ruler4k_m5_conformal_auto",
        "qwen8b_ruler8k_m5_conformal_auto",
        "qwen8b_ruler4k_m5_var_minsafe_tail035",
        "qwen8b_ruler8k_m5_var_minsafe_tail035",
        "qwen8b_ruler4k_m5_conformal_floor2",
        "qwen8b_ruler8k_m5_conformal_floor2",
        "qwen8b_ruler16k_m3_conformal_auto_sharded",
    ]:
        require(experiment_name in summarizer.EXPERIMENTS, f"summarizer missing {experiment_name}", errors)

    table_script = PROJECT_ROOT / "scripts/make_icml_runtime_tables_20260707.py"
    require(table_script.exists(), "missing ICML table generation script", errors)
    paper_table_script = PROJECT_ROOT / "scripts/make_icml_paper_tables_20260707.py"
    require(paper_table_script.exists(), "missing ICML paper table generation script", errors)
    paper_table_text = paper_table_script.read_text(encoding="utf-8")
    require("paper_main_table.md" in paper_table_text, "paper table script missing main table output", errors)
    require("paper_floor_ablation.md" in paper_table_text, "paper table script missing floor ablation output", errors)
    require("paper_rope_ablation.md" in paper_table_text, "paper table script missing RoPE ablation output", errors)
    figure_script = PROJECT_ROOT / "scripts/plot_icml_runtime_figures_20260707.py"
    require(figure_script.exists(), "missing ICML figure generation script", errors)
    readiness_script = PROJECT_ROOT / "scripts/make_icml_readiness_report_20260707.py"
    require(readiness_script.exists(), "missing ICML readiness report script", errors)
    readiness_text = readiness_script.read_text(encoding="utf-8")
    require("LongBench input planner" in readiness_text, "readiness report missing LongBench gate", errors)
    require("RULER scaling coverage" in readiness_text, "readiness report missing scaling gate", errors)
    overhead_script = PROJECT_ROOT / "scripts/make_icml_overhead_report_20260707.py"
    require(overhead_script.exists(), "missing ICML overhead report script", errors)
    overhead_text = overhead_script.read_text(encoding="utf-8")
    require("net_component_gain_ms" in overhead_text, "overhead report missing net component gain", errors)
    runtime_control_test = PROJECT_ROOT / "scripts/test_runtime_controls_20260707.py"
    require(runtime_control_test.exists(), "missing runtime control test script", errors)

    collect_script = PROJECT_ROOT / "scripts/collect_icml_runtime_status_20260707.ps1"
    collect_text = collect_script.read_text(encoding="utf-8")
    require("make_icml_runtime_tables_20260707.py" in collect_text, "collect script does not generate ICML tables", errors)
    require("make_icml_paper_tables_20260707.py" in collect_text, "collect script does not generate ICML paper tables", errors)
    require("plot_icml_runtime_figures_20260707.py" in collect_text, "collect script does not generate ICML figures", errors)
    require("make_icml_readiness_report_20260707.py" in collect_text, "collect script does not generate readiness report", errors)
    require("make_icml_overhead_report_20260707.py" in collect_text, "collect script does not generate overhead report", errors)

    output_verifier_scripts = [
        "scripts/run_output_verifier_floor_sweep_20260707.sh",
        "scripts/run_ruler_scaling_expansion_20260707.sh",
        "scripts/run_ruler16k_case1_shards_20260707.sh",
        "scripts/run_ruler16k_floor2_recovery_20260707.sh",
    ]
    for rel_path in output_verifier_scripts:
        text = (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")
        require(
            "--runtime_methods full_kv_cache,output_level_risk_kv_planner" in text,
            f"{rel_path} is not method-filtered",
            errors,
        )

    if errors:
        print("manifest audit FAILED")
        for item in errors:
            print(f"- {item}")
        raise SystemExit(1)

    print("manifest audit OK")
    print(f"primary_experiments={len(manifest['primary_experiments'])}")
    print(f"pending_scaling_experiments={len(manifest['pending_scaling_experiments'])}")
    print(f"summarized_experiments={len(summarizer.EXPERIMENTS)}")


if __name__ == "__main__":
    main()
