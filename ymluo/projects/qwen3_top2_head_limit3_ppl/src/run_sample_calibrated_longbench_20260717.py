from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from countcap_cost_gate_20260723 import choose_countcap_path, load_cost_profile
from preallocated_dynamic_cache_20260724 import PreallocatedDynamicCache
import run_controlled_public_kv_benchmark_v1 as lb
from run_critical_position_budget_probe_20260715 import (
    run_one_token,
    summarize_attention_records,
)
from run_head_top2_targeted_ppl_20260714 import (
    capture_qk_trace,
    export_active_keypca_basis_templates,
    head_adaptive_mass_mode,
    head_qabs_sampled_mass_mode,
    head_top_fraction_mode,
    install_active_binarypc_projections,
    install_active_keypca_basis_templates,
    install_active_packed_qmse_templates,
    install_llama_head_top_fraction_patch,
    load_model,
    precompute_active_packed_qmse_qk_factors,
    precompute_qksieve_value_metric_grams,
    prebuild_active_keypca_int4_index,
    prefill_query_moment_mode,
    prefill_query_tail_mode,
    preload_qksieve_qmse_rate_tables,
    preload_qksieve_runtime_extensions,
    seed_packed_qmse_prefill_queries,
    seed_rabitq_prefill_query_moments,
    set_attention_implementation,
)


DEFAULT_TASKS = "narrativeqa,hotpotqa,passage_retrieval_en,lcc"
FROZEN_BUDGET_FRACTIONS = (0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08)
COUNTCAP_SCORE_MODE = (
    "pca_int4_chunked_logscale16_qkmetric_sampleq_autosplit"
)
COUNTCAP_KEYPCA_SCORE_MODE = "pca_int4_chunked_logscale16_sampleq_autosplit"
COUNTCAP_KEYPCA_DIRECT_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_autosplit"
)
COUNTCAP_KEYPCA_DIRECT_FUSED_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_fused_autosplit"
)
COUNTCAP_QK_BALANCED_PACKED_METHOD = (
    "countcap_fullprompt_qkbalanced_packed_direct"
)
QKSIEVE_FULLTOPK_METHOD = (
    "qksieve_fullprompt_auto_plain_fulltopk"
)
QKSIEVE_MEANVALUE_FULLTOPK_METHOD = (
    "qksieve_fullprompt_auto_plain_meanvalue_fulltopk"
)
QKSIEVE_QFUSED_FULLTOPK_METHOD = (
    "qksieve_fullprompt_auto_plain_qfused_fulltopk"
)
QKSIEVE_FULLTOPK_FP16_METHOD = (
    "qksieve_fullprompt_auto_plain_fulltopk_fp16"
)
QKSIEVE_FIXED410_FULLTOPK_METHOD = (
    "qksieve_fullprompt_fixed410_fulltopk"
)
QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_METHOD = (
    "qksieve_fullprompt_fixed410_post2xprererank_l00to08_fulltopk"
)
FIER_RTN1_G32_FULLTOPK_METHOD = "fier_rtn1_g32_fulltopk"
FIER_RTN1_G32_PACKED_FULLTOPK_METHOD = (
    "fier_rtn1_g32_packed_fulltopk"
)
QUEST_P16_FULLTOPK_METHOD = "quest_p16_fullprompt_matchedbudget"
UNIQUE_P8_FULLTOPK_METHOD = "unique_p8_fullprompt_matchedbudget"
RABITQCACHE_RTN1_FULLTOPK_METHOD = (
    "rabitqcache_rtn1_fullprompt_matchedbudget"
)
BINARYPC_OFFLINE64_FULLTOPK_METHOD = (
    "binarypc_offline64_fullprompt_matchedbudget"
)
SPARQ_R32_SELECTOR_FULLTOPK_METHOD = (
    "sparq_r32_selector_fullprompt_matchedbudget"
)
SPARQ_R32_FORMULA_FULLTOPK_METHOD = (
    "sparq_r32_formula_fullprompt_matchedbudget"
)
QKSIEVE_KEYPCA_UNIFORM1_FULLTOPK_METHOD = (
    "qksieve_fullprompt_keypca_uniform1_fulltopk"
)
QKSIEVE_QKBALANCED_UNIFORM1_FULLTOPK_METHOD = (
    "qksieve_fullprompt_qkbalanced_uniform1_fulltopk"
)
QKSIEVE_RANDOM_UNIFORM1_FULLTOPK_METHOD = (
    "qksieve_fullprompt_random_uniform1_fulltopk"
)
QKSIEVE_KEYPCA_AUTOKEY_FULLTOPK_METHOD = (
    "qksieve_fullprompt_keypca_autokey_fulltopk"
)
QKSIEVE_QKBALANCED_AUTOKEY_FULLTOPK_METHOD = (
    "qksieve_fullprompt_qkbalanced_autokey_fulltopk"
)
QKSIEVE_GLOBAL_WMMA_SAMPLED_METHOD = (
    "qksieve_global_qkbalanced_qmse_wmma_sampled"
)
QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLED_METHOD = (
    "qksieve_global_qkbalanced_keymse_wmma_sampled"
)
QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLEMASS_METHOD = (
    "qksieve_global_qkbalanced_keymse_wmma_samplemass"
)
QKSIEVE_GLOBAL_KEYMSE_WMMA_PROXYMASS_METHOD = (
    "qksieve_global_qkbalanced_keymse_wmma_proxymass"
)
QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_SAMPLED_METHOD = (
    "qksieve_requestlocal_qkbalanced_keymse_wmma_sampled"
)
QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_PROXYMASS_METHOD = (
    "qksieve_requestlocal_qkbalanced_keymse_wmma_proxymass"
)
QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_MEANTAIL_METHOD = (
    "qksieve_requestlocal_qkbalanced_keymse_wmma_meantail"
)
QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH8_METHOD = (
    "qksieve_requestlocal_qkbalanced_keymse_wmma_valuesketch8"
)
QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH16_METHOD = (
    "qksieve_requestlocal_qkbalanced_keymse_wmma_valuesketch16"
)
QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH32_METHOD = (
    "qksieve_requestlocal_qkbalanced_keymse_wmma_valuesketch32"
)
QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH_PROGRESSIVE_METHOD = (
    "qksieve_requestlocal_qkbalanced_keymse_wmma_valuesketch8to32"
)
QKSIEVE_FROZEN_C64_METHOD = (
    "qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64"
)
QKSIEVE_GLOBAL_WMMA_SAMPLED_METHODS = {
    QKSIEVE_GLOBAL_WMMA_SAMPLED_METHOD,
    QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLED_METHOD,
    QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLEMASS_METHOD,
    QKSIEVE_GLOBAL_KEYMSE_WMMA_PROXYMASS_METHOD,
}
QKSIEVE_WMMA_SAMPLED_METHODS = {
    *QKSIEVE_GLOBAL_WMMA_SAMPLED_METHODS,
    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_SAMPLED_METHOD,
    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_PROXYMASS_METHOD,
    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_MEANTAIL_METHOD,
    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH8_METHOD,
    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH16_METHOD,
    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH32_METHOD,
    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH_PROGRESSIVE_METHOD,
    QKSIEVE_FROZEN_C64_METHOD,
}
COUNTCAP_QK_BALANCED_FIXED4421_PACKED_METHOD = (
    "countcap_fullprompt_qkbalanced_fixed4421_packed_direct"
)
COUNTCAP_QK_BALANCED_QSCALE_PACKED_METHOD = (
    "countcap_fullprompt_qkbalanced_qscale_packed_direct"
)
COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_PACKED_METHOD = (
    "countcap_fullprompt_qkbalanced_fixed4421_qscale_packed_direct"
)
COUNTCAP_QK_BALANCED_QSCALE_OAS_PACKED_METHOD = (
    "countcap_fullprompt_qkbalanced_qscale_oas_packed_direct"
)
COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_OAS_PACKED_METHOD = (
    "countcap_fullprompt_qkbalanced_fixed4421_qscale_oas_packed_direct"
)
COUNTCAP_QK_BALANCED_CENTERED_PACKED_METHOD = (
    "countcap_fullprompt_qkbalanced_centered_packed_direct"
)
COUNTCAP_QK_BALANCED_SHAREDTAIL_PACKED_METHOD = (
    "countcap_fullprompt_qkbalanced_sharedtail240_centered_packed_direct"
)
COUNTCAP_QK_BALANCED_PACKED_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_direct"
)
QKSIEVE_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk"
)
QKSIEVE_MEANVALUE_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "meanvalue_packed_fulltopk"
)
QKSIEVE_QFUSED_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_packed_fulltopk"
)
QKSIEVE_FULLTOPK_FP16_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk_fp16"
)
QKSIEVE_FIXED410_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_packed_fulltopk"
)
QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_"
    "post2xprererank_l00to08_packed_fulltopk"
)
FIER_RTN1_G32_FULLTOPK_SCORE_MODE = "fier_rtn1_g32_fulltopk"
FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE = (
    "fier_rtn1_g32_packed_fulltopk"
)
QUEST_P16_FULLTOPK_SCORE_MODE = "quest_p16_fulltopk"
UNIQUE_P8_FULLTOPK_SCORE_MODE = "unique_p8_meanstd_fulltopk"
RABITQCACHE_RTN1_FULLTOPK_SCORE_MODE = "rabitqcache_rtn1_fulltopk"
BINARYPC_OFFLINE64_FULLTOPK_SCORE_MODE = "binarypc_offline64_fulltopk"
SPARQ_R32_SELECTOR_FULLTOPK_SCORE_MODE = (
    "sparq_r32_selector_fulltopk"
)
SPARQ_R32_FORMULA_FULLTOPK_SCORE_MODE = (
    "sparq_r32_meanvalue_fulltopk"
)


def configured_index_bits_per_token(score_mode: str) -> float:
    """Return the exact static index-rate contract, not a runtime sample."""
    if score_mode == QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH8_SCORE_MODE:
        return 273.0
    if score_mode in {
        QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH16_SCORE_MODE,
        QKSIEVE_FROZEN_C64_SCORE_MODE,
    }:
        return 306.0
    if score_mode == QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH32_SCORE_MODE:
        return 372.0
    if score_mode == QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH_PROGRESSIVE_SCORE_MODE:
        return 372.0
    if score_mode in {
        QKSIEVE_FIXED410_FULLTOPK_SCORE_MODE,
        QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_SCORE_MODE,
    }:
        return 112.0
    if score_mode in {
        QKSIEVE_FULLTOPK_SCORE_MODE,
        QKSIEVE_MEANVALUE_FULLTOPK_SCORE_MODE,
        QKSIEVE_QFUSED_FULLTOPK_SCORE_MODE,
        QKSIEVE_FULLTOPK_FP16_SCORE_MODE,
        QKSIEVE_GLOBAL_WMMA_SAMPLED_SCORE_MODE,
        QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLED_SCORE_MODE,
        QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLEMASS_SCORE_MODE,
        QKSIEVE_GLOBAL_KEYMSE_WMMA_PROXYMASS_SCORE_MODE,
        QKSIEVE_GLOBAL_KEYMSE_WMMA_MEANTAIL_SCORE_MODE,
        COUNTCAP_QK_BALANCED_PACKED_SCORE_MODE,
    }:
        return 240.0
    if score_mode in {
        FIER_RTN1_G32_FULLTOPK_SCORE_MODE,
        FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE,
        QUEST_P16_FULLTOPK_SCORE_MODE,
    }:
        return 256.0
    if score_mode == UNIQUE_P8_FULLTOPK_SCORE_MODE:
        return 258.0
    if score_mode == RABITQCACHE_RTN1_FULLTOPK_SCORE_MODE:
        return 224.0
    if score_mode == BINARYPC_OFFLINE64_FULLTOPK_SCORE_MODE:
        return 64.0
    if score_mode in {
        SPARQ_R32_SELECTOR_FULLTOPK_SCORE_MODE,
        SPARQ_R32_FORMULA_FULLTOPK_SCORE_MODE,
    }:
        return 0.0
    return 0.0


def uses_dense_prompt_suffix(method: str) -> bool:
    """Return whether a method consumes the prompt suffix as dense prefill."""

    return (
        method.startswith("countcap_fullprompt")
        or method.startswith("qksieve_fullprompt")
        or method in QKSIEVE_WMMA_SAMPLED_METHODS
        or method
        in {
            FIER_RTN1_G32_FULLTOPK_METHOD,
            FIER_RTN1_G32_PACKED_FULLTOPK_METHOD,
            QUEST_P16_FULLTOPK_METHOD,
            UNIQUE_P8_FULLTOPK_METHOD,
            RABITQCACHE_RTN1_FULLTOPK_METHOD,
            BINARYPC_OFFLINE64_FULLTOPK_METHOD,
            SPARQ_R32_SELECTOR_FULLTOPK_METHOD,
            SPARQ_R32_FORMULA_FULLTOPK_METHOD,
        }
        or method == "countcap_massadaptive_fullprompt"
    )
QKSIEVE_KEYPCA_UNIFORM1_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed11111111_packed_fulltopk"
)
QKSIEVE_QKBALANCED_UNIFORM1_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed11111111_qkmetric_packed_fulltopk"
)
QKSIEVE_RANDOM_UNIFORM1_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed11111111_random_packed_fulltopk"
)
QKSIEVE_KEYPCA_AUTOKEY_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_packed_fulltopk"
)
QKSIEVE_QKBALANCED_AUTOKEY_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_packed_fulltopk"
)
QKSIEVE_GLOBAL_WMMA_SAMPLED_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_unbiased_packed_direct"
)
QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLED_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_unbiased_packed_direct"
)
QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLEMASS_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_samplemass_unbiased_packed_direct"
)
QKSIEVE_GLOBAL_KEYMSE_WMMA_PROXYMASS_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_proxymass_unbiased_packed_direct"
)
QKSIEVE_GLOBAL_KEYMSE_WMMA_MEANTAIL_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_meantail_unbiased_packed_direct"
)
QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH8_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch8i4shared_unbiased_packed_direct"
)
QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH16_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch16i4shared_unbiased_packed_direct"
)
QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH32_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch32i4shared_unbiased_packed_direct"
)
QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH_PROGRESSIVE_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch8to32i4shared_tau40m_unbiased_packed_direct"
)
QKSIEVE_FROZEN_C64_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch16i4shared_wometric_sortcompact_unbiased_"
    "packed_direct_oas"
)
QKSIEVE_GLOBAL_WMMA_SAMPLED_SCORE_MODES = {
    QKSIEVE_GLOBAL_WMMA_SAMPLED_SCORE_MODE,
    QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLED_SCORE_MODE,
    QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLEMASS_SCORE_MODE,
    QKSIEVE_GLOBAL_KEYMSE_WMMA_PROXYMASS_SCORE_MODE,
    QKSIEVE_GLOBAL_KEYMSE_WMMA_MEANTAIL_SCORE_MODE,
    QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH8_SCORE_MODE,
    QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH16_SCORE_MODE,
    QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH32_SCORE_MODE,
    QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH_PROGRESSIVE_SCORE_MODE,
    QKSIEVE_FROZEN_C64_SCORE_MODE,
}
COUNTCAP_QK_BALANCED_FIXED4421_PACKED_SCORE_MODE = (
    "pca_hierarchical_fixed4421_qkmetric_packed_direct"
)
COUNTCAP_QK_BALANCED_QSCALE_PACKED_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qscale_packed_direct"
)
COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_PACKED_SCORE_MODE = (
    "pca_hierarchical_fixed4421_qkmetric_qscale_packed_direct"
)
COUNTCAP_QK_BALANCED_QSCALE_OAS_PACKED_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qscale_oas_packed_direct"
)
COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_OAS_PACKED_SCORE_MODE = (
    "pca_hierarchical_fixed4421_qkmetric_qscale_oas_packed_direct"
)
COUNTCAP_QK_BALANCED_CENTERED_PACKED_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_centered_packed_direct"
)
COUNTCAP_QK_BALANCED_SHAREDTAIL_PACKED_SCORE_MODE = (
    "pca_hierarchical_sharedtail240_qkmetric_centered_packed_direct"
)
PACKED_QUERY_CALIBRATED_SCORE_MODES = frozenset(
    {
        COUNTCAP_QK_BALANCED_PACKED_SCORE_MODE,
        QKSIEVE_FULLTOPK_SCORE_MODE,
        QKSIEVE_MEANVALUE_FULLTOPK_SCORE_MODE,
        QKSIEVE_QFUSED_FULLTOPK_SCORE_MODE,
        QKSIEVE_FULLTOPK_FP16_SCORE_MODE,
        QKSIEVE_FIXED410_FULLTOPK_SCORE_MODE,
        QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_SCORE_MODE,
        QKSIEVE_QKBALANCED_UNIFORM1_FULLTOPK_SCORE_MODE,
        QKSIEVE_QKBALANCED_AUTOKEY_FULLTOPK_SCORE_MODE,
        *QKSIEVE_GLOBAL_WMMA_SAMPLED_SCORE_MODES,
        COUNTCAP_QK_BALANCED_FIXED4421_PACKED_SCORE_MODE,
        COUNTCAP_QK_BALANCED_QSCALE_PACKED_SCORE_MODE,
        COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_PACKED_SCORE_MODE,
        COUNTCAP_QK_BALANCED_QSCALE_OAS_PACKED_SCORE_MODE,
        COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_OAS_PACKED_SCORE_MODE,
        COUNTCAP_QK_BALANCED_CENTERED_PACKED_SCORE_MODE,
        COUNTCAP_QK_BALANCED_SHAREDTAIL_PACKED_SCORE_MODE,
    }
)
COUNTCAP_KEYPCA_SCAN_QK_FUSED_AV_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_scanqk_fusedav"
)
COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_qkvfused"
)
COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_DP4A_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_dp4a"
)
COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_QPROJ_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qproj"
)
COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_QPROJSCAN_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qprojscan"
)
COUNTCAP_QPROJSCAN_SPLIT_METHODS = {
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplit2_prefillindex": (
        "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qprojscan_qkvsplit2"
    ),
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplit4_prefillindex": (
        "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qprojscan_qkvsplit4"
    ),
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_prefillindex": (
        "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qprojscan_qkvsplitauto"
    ),
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_inplacecache_prefillindex": (
        "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qprojscan_qkvsplitauto"
    ),
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_cacheauto_prefillindex": (
        "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qprojscan_qkvsplitauto"
    ),
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_cacheauto_reuse2_prefillindex": (
        "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qprojscan_qkvsplitauto_reuse2"
    ),
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_cacheauto_reuse4_prefillindex": (
        "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qprojscan_qkvsplitauto_reuse4"
    ),
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_cacheauto_reuse8_prefillindex": (
        "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qprojscan_qkvsplitauto_reuse8"
    ),
}
COUNTCAP_CACHE_AUTO_METHOD = (
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_"
    "qkvsplitauto_cacheauto_prefillindex"
)
COUNTCAP_CACHE_AUTO_DEFAULT_MIN_TOKENS = 14_000
COUNTCAP_DIRECT_FRACTION = 0.06
COUNTCAP_DIRECT_MIN_TOKENS = 256
COUNTCAP_DIRECT_MAX_TOKENS = 1_280
COUNTCAP_TEMPORAL_MASS_GATE_THRESHOLDS = {
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_massgate90_prefillindex": 0.90,
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_massgate94_prefillindex": 0.94,
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_massgate95_prefillindex": 0.95,
}
COUNTCAP_TEMPORAL_GQA_MASS_GATE_THRESHOLDS = {
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_gqamassgate94_prefillindex": 0.94,
    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_gqamassgate95_prefillindex": 0.95,
}


def should_use_preallocated_cache(method: str, prompt_tokens: int) -> bool:
    if method in {
        "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_inplacecache_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_inplacecache_prefillindex",
    }:
        return True
    if (
        method != COUNTCAP_CACHE_AUTO_METHOD
        and "_qkvsplitauto_cacheauto_reuse" not in method
    ):
        return False
    min_tokens = int(
        os.environ.get(
            "COUNTCAP_INPLACE_CACHE_MIN_TOKENS",
            str(COUNTCAP_CACHE_AUTO_DEFAULT_MIN_TOKENS),
        )
    )
    return int(prompt_tokens) >= min_tokens
COUNTCAP_TEMPORAL_MASS_GATE_METHODS = {
    **COUNTCAP_TEMPORAL_MASS_GATE_THRESHOLDS,
    **COUNTCAP_TEMPORAL_GQA_MASS_GATE_THRESHOLDS,
}
COUNTCAP_KEYPCA_DIRECT_QKV_SPLIT4_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_qkvsplit4"
)
COUNTCAP_KEYPCA_DIRECT_PROXY_AV_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_proxyav"
)
COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_REUSE_SCORE_MODES = {
    2: "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_reuse2",
    4: "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_reuse4",
    8: "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_reuse8",
}
COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_GQA_UNION_REUSE2_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_gqaunion_reuse2"
)
COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_STABLE16_REUSE2_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_stable16_reuse2"
)
COUNTCAP_KEYPCA_DIRECT_QKV_WARP_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_qkvwarp"
)
COUNTCAP_ADAPTIVE_SCORE_MODE = "pca_int4_chunked_logscale16_autosplit"
COUNTCAP_PROJECTION_DIM = 48
COUNTCAP_MAX_ATTENTION_TOKENS = 1280
COUNTCAP_ADAPTIVE_BUDGET_FRACTIONS = (0.02, 0.03, 0.04, 0.06)
TEMPORAL_TRACE_METRICS = (
    "periodic_candidate_reuse_rate",
    "temporal_trace_available",
    "temporal_candidate_jaccard",
    "temporal_candidate_recall_from_previous",
    "temporal_candidate_recall_from_previous_with_new_tokens",
    "temporal_candidate_jaccard_with_new_tokens",
    "temporal_query_delta_norm",
    "temporal_key_norm_bound",
    "temporal_score_change_bound",
    "temporal_boundary_margin",
    "temporal_threshold_slack",
    "temporal_rejected_max_score",
    "temporal_certificate_safe",
    "temporal_certificate_layer_safe",
    "temporal_certificate_margin_ratio",
    "temporal_expected_score_delta_std",
    "temporal_core_margin_top25",
    "temporal_core_margin_ratio_top25",
    "temporal_core_recall_from_previous_top25",
    "temporal_expected_core_crossings",
    "temporal_expected_core_crossing_fraction",
    "temporal_reuse_output_trace_available",
    "temporal_reuse_output_relative_error",
    "temporal_reuse_output_cosine",
    "temporal_reuse_fresh_attention_mass",
    "temporal_reuse_output_error_le_1pct",
    "temporal_reuse_output_error_le_2pct",
    "temporal_reuse_mass_ge_95pct",
    "temporal_reuse_mass_ge_99pct",
    "temporal_sampled_reuse_mass_estimate",
    "temporal_sampled_reuse_mass_absolute_error",
    "temporal_gqa_union_candidate_fraction",
    "temporal_gqa_union_fresh_attention_mass",
    "temporal_gqa_union_output_relative_error",
    "temporal_gqa_union_output_error_le_1pct",
    "temporal_gqa_union_output_error_le_2pct",
    "temporal_sampled_safe_s90",
    "temporal_sampled_safe_s90_bad_output",
    "temporal_sampled_safe_s90_low_mass",
    "temporal_sampled_safe_s95",
    "temporal_sampled_safe_s95_bad_output",
    "temporal_sampled_safe_s95_low_mass",
    "temporal_sampled_safe_s97",
    "temporal_sampled_safe_s97_bad_output",
    "temporal_sampled_safe_s97_low_mass",
    "temporal_sampled_safe_s99",
    "temporal_sampled_safe_s99_bad_output",
    "temporal_sampled_safe_s99_low_mass",
    "temporal_expected_safe_r0p25",
    "temporal_expected_safe_r0p25_bad_output",
    "temporal_expected_safe_r0p25_low_mass",
    "temporal_expected_safe_r0p5",
    "temporal_expected_safe_r0p5_bad_output",
    "temporal_expected_safe_r0p5_low_mass",
    "temporal_expected_safe_r1",
    "temporal_expected_safe_r1_bad_output",
    "temporal_expected_safe_r1_low_mass",
    "temporal_expected_safe_r2",
    "temporal_expected_safe_r2_bad_output",
    "temporal_expected_safe_r2_low_mass",
    "temporal_core_safe_m0p5",
    "temporal_core_safe_m0p5_bad_output",
    "temporal_core_safe_m0p5_low_mass",
    "temporal_core_safe_m1",
    "temporal_core_safe_m1_bad_output",
    "temporal_core_safe_m1_low_mass",
    "temporal_core_safe_m2",
    "temporal_core_safe_m2_bad_output",
    "temporal_core_safe_m2_low_mass",
    "temporal_core_safe_m4",
    "temporal_core_safe_m4_bad_output",
    "temporal_core_safe_m4_low_mass",
    "temporal_mass_gate_reuse_rate",
    "temporal_mass_gate_estimated_mass",
    "temporal_mass_gate_candidate_fraction",
    "temporal_mass_gate_reused_output_relative_error",
    "temporal_mass_gate_bad_output",
    "temporal_mass_gate_cache_age",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aligned FullKV versus sample-calibrated global partition retrieval "
            "on LongBench."
        )
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--longbench_data_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument(
        "--methods", default="full_kv,global_partition,qgate_partition"
    )
    parser.add_argument("--max_samples_per_task", type=int, default=5)
    parser.add_argument(
        "--sample_offset_per_task",
        type=int,
        default=0,
        help="Skip this many source rows before applying the per-task sample limit.",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_context_tokens", type=int, default=7500)
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=0,
        help=(
            "When positive, apply the selected prompt truncation protocol."
        ),
    )
    parser.add_argument(
        "--prompt_truncation_mode",
        choices=("preserve_suffix", "official_middle"),
        default="preserve_suffix",
        help=(
            "preserve_suffix keeps the complete task/query suffix; "
            "official_middle reproduces KVCache-Factory whole-prompt "
            "first/last-half tokenization before chat wrapping"
        ),
    )
    parser.add_argument(
        "--official_query_tail_tokens",
        type=int,
        default=8,
        help=(
            "Dense final prompt tokens used to calibrate QK-balanced "
            "retrieval under official_middle."
        ),
    )
    parser.add_argument("--max_new_tokens_override", type=int, default=64)
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument(
        "--prompt_wrapper",
        choices=("llama3", "qwen3", "tokenizer_chat", "none"),
        default="llama3",
    )
    parser.add_argument("--minimum_sparse_prefix_tokens", type=int, default=0)
    parser.add_argument("--collect_attention_stats", action="store_true")
    parser.add_argument("--mass_threshold", type=float, default=0.75)
    parser.add_argument(
        "--budget_fractions",
        default=",".join(str(value) for value in FROZEN_BUDGET_FRACTIONS),
    )
    parser.add_argument("--sample_fraction", type=float, default=0.0025)
    parser.add_argument("--candidate_fraction", type=float, default=0.08)
    parser.add_argument(
        "--countcap_direct_fraction_override",
        type=float,
        default=0.0,
        help="Optional experiment-only override for direct CountCap candidates.",
    )
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument(
        "--qk_metric_query_shrinkage",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--packed_qmse_template_in",
        type=Path,
        default=None,
        help=(
            "Model-level QKSieve template required by the global WMMA "
            "sampled-quantile method."
        ),
    )
    parser.add_argument(
        "--binarypc_projection_path",
        type=Path,
        default=None,
        help=(
            "Official BinaryPC offline projection checkpoint required by "
            "binarypc_offline64_fullprompt_matchedbudget."
        ),
    )
    parser.add_argument(
        "--sampled_quantile_sample_count",
        type=int,
        default=256,
        help="Proxy scores sampled per query head to estimate the selection threshold.",
    )
    parser.add_argument(
        "--sampled_quantile_target_tail_count",
        type=int,
        default=0,
        help=(
            "If positive, replace the fixed sample count by the smallest "
            "supported multiple of 256 whose expected target-tail count "
            "is at least this value."
        ),
    )
    parser.add_argument("--partition_ucb_z", type=float, default=0.0)
    parser.add_argument("--partition_overfetch_factor", type=int, default=2)
    parser.add_argument("--value_mass_threshold", type=float, default=1.0)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_memory_per_gpu_gib", type=float, default=0.0)
    parser.add_argument("--cost_profile", type=Path)
    parser.add_argument("--cost_quality_floor", type=float, default=0.95)
    parser.add_argument("--cost_speed_margin", type=float, default=1.03)
    parser.add_argument(
        "--cost_expected_generation_tokens",
        type=int,
        default=0,
        help="Expected generation horizon for countcap_auto; zero uses the task limit.",
    )
    parser.add_argument(
        "--qk_trace_output_dir",
        type=Path,
        default=None,
        help="Optional directory for sparse generation-position Q/K traces.",
    )
    parser.add_argument(
        "--qk_trace_method",
        default=QKSIEVE_FULLTOPK_METHOD,
    )
    parser.add_argument(
        "--qk_trace_layers",
        default="0,8,16,24,31",
    )
    parser.add_argument(
        "--qk_trace_steps",
        default="0,1,3,7,15,31,63,127,255,511,1023,2047,4095",
    )
    parser.add_argument(
        "--qk_trace_prefill_query_tail_tokens",
        type=int,
        default=32,
        help=(
            "Prompt-tail Query positions exported for sample-count analysis. "
            "This does not change --official_query_tail_tokens used by the "
            "frozen retrieval method."
        ),
    )
    return parser.parse_args()


def parse_csv_values(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def parse_nonnegative_ints(spec: str, name: str) -> tuple[int, ...]:
    values = tuple(
        sorted({int(item) for item in spec.split(",") if item.strip()})
    )
    if not values or values[0] < 0:
        raise ValueError(f"{name} must contain non-negative integers")
    return values


def resolve_qk_trace_config(
    args: argparse.Namespace,
    methods: list[str],
) -> dict[str, Any] | None:
    output_dir = getattr(args, "qk_trace_output_dir", None)
    if output_dir is None:
        return None
    method = str(args.qk_trace_method)
    if method not in methods:
        raise ValueError("qk_trace_method must be included in --methods")
    if not method.startswith("qksieve_fullprompt"):
        raise ValueError("Q/K drift tracing is restricted to QKSieve methods")
    layers = parse_nonnegative_ints(args.qk_trace_layers, "qk_trace_layers")
    steps = parse_nonnegative_ints(args.qk_trace_steps, "qk_trace_steps")
    if steps[0] != 0:
        raise ValueError("qk_trace_steps must include decode step 0")
    record_tail_tokens = int(
        getattr(args, "qk_trace_prefill_query_tail_tokens", 32)
    )
    if record_tail_tokens <= 0:
        raise ValueError(
            "qk_trace_prefill_query_tail_tokens must be positive"
        )
    return {
        "output_dir": Path(output_dir),
        "method": method,
        "layers": layers,
        "steps": steps,
        "prefill_query_tail_tokens": record_tail_tokens,
    }


def qk_trace_path(
    output_dir: Path,
    task: str,
    sample_id: str,
    method: str,
) -> Path:
    safe = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{task}__{sample_id}__{method}",
    ).strip("._")
    if not safe:
        raise ValueError("trace filename is empty after sanitization")
    return output_dir / f"{safe}.pt"


def parse_methods(spec: str) -> list[str]:
    methods = parse_csv_values(spec)
    allowed = {
        "full_kv",
        "global_partition",
        "qgate_partition",
        "ec_bandef",
        "ec_bandef_budget",
        "oneshot_bandef_budget",
        "countcap",
        "countcap_fullprompt",
        "countcap_fullprompt_keypca",
        "countcap_fullprompt_keypca_direct",
        "countcap_fullprompt_keypca_direct_fused",
        COUNTCAP_QK_BALANCED_PACKED_METHOD,
        QKSIEVE_FULLTOPK_METHOD,
        QKSIEVE_MEANVALUE_FULLTOPK_METHOD,
        QKSIEVE_QFUSED_FULLTOPK_METHOD,
        QKSIEVE_FULLTOPK_FP16_METHOD,
        QKSIEVE_FIXED410_FULLTOPK_METHOD,
        QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_METHOD,
        FIER_RTN1_G32_FULLTOPK_METHOD,
        FIER_RTN1_G32_PACKED_FULLTOPK_METHOD,
        QUEST_P16_FULLTOPK_METHOD,
        UNIQUE_P8_FULLTOPK_METHOD,
        RABITQCACHE_RTN1_FULLTOPK_METHOD,
        BINARYPC_OFFLINE64_FULLTOPK_METHOD,
        SPARQ_R32_SELECTOR_FULLTOPK_METHOD,
        SPARQ_R32_FORMULA_FULLTOPK_METHOD,
        QKSIEVE_KEYPCA_UNIFORM1_FULLTOPK_METHOD,
        QKSIEVE_QKBALANCED_UNIFORM1_FULLTOPK_METHOD,
        QKSIEVE_RANDOM_UNIFORM1_FULLTOPK_METHOD,
        QKSIEVE_KEYPCA_AUTOKEY_FULLTOPK_METHOD,
        QKSIEVE_QKBALANCED_AUTOKEY_FULLTOPK_METHOD,
        QKSIEVE_GLOBAL_WMMA_SAMPLED_METHOD,
        QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLED_METHOD,
        QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLEMASS_METHOD,
        QKSIEVE_GLOBAL_KEYMSE_WMMA_PROXYMASS_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_SAMPLED_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_PROXYMASS_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_MEANTAIL_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH8_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH16_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH32_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH_PROGRESSIVE_METHOD,
        QKSIEVE_FROZEN_C64_METHOD,
        COUNTCAP_QK_BALANCED_FIXED4421_PACKED_METHOD,
        COUNTCAP_QK_BALANCED_QSCALE_PACKED_METHOD,
        COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_PACKED_METHOD,
        COUNTCAP_QK_BALANCED_QSCALE_OAS_PACKED_METHOD,
        COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_OAS_PACKED_METHOD,
        COUNTCAP_QK_BALANCED_CENTERED_PACKED_METHOD,
        COUNTCAP_QK_BALANCED_SHAREDTAIL_PACKED_METHOD,
        "countcap_fullprompt_keypca_scanqk_fusedav_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused",
        "countcap_fullprompt_keypca_direct_qkvfused_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_asyncprefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_dp4a_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_qproj_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_inplacecache_prefillindex",
        *COUNTCAP_QPROJSCAN_SPLIT_METHODS,
        *COUNTCAP_TEMPORAL_MASS_GATE_METHODS,
        "countcap_fullprompt_keypca_direct_qkvsplit4_prefillindex",
        "countcap_fullprompt_keypca_direct_proxyav_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_reuse2_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_reuse4_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_reuse8_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_gqaunion_reuse2_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_stable16_reuse2_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvwarp",
        "exact_top2_fullprompt",
        "exact_massadaptive_fullprompt",
        "countcap_massadaptive_fullprompt",
        "countcap_auto",
    }
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"unsupported methods: {unknown}")
    if not methods:
        raise ValueError("at least one method is required")
    return list(dict.fromkeys(methods))


def parse_budget_fractions(spec: str) -> tuple[float, ...]:
    fractions = tuple(sorted({float(item) for item in parse_csv_values(spec)}))
    if not fractions or fractions[0] <= 0.0 or fractions[-1] > 1.0:
        raise ValueError("budget fractions must be in (0, 1]")
    return fractions


def countcap_config(
    history_tokens: int,
    score_mode: str = COUNTCAP_SCORE_MODE,
) -> dict[str, Any]:
    """Return the frozen numerical CountCap policy for one prompt length."""
    if history_tokens <= 0:
        raise ValueError("history_tokens must be positive")
    attention_tokens = min(
        max(1, round(0.02 * history_tokens)),
        COUNTCAP_MAX_ATTENTION_TOKENS,
    )
    attention_fraction = attention_tokens / history_tokens
    candidate_fraction = min(0.06, max(0.03, 4.0 * attention_fraction))
    return {
        "budget_fractions": (attention_fraction,),
        "candidate_fraction": candidate_fraction,
        "projection_dim": COUNTCAP_PROJECTION_DIM,
        "score_mode": score_mode,
        "attention_tokens": attention_tokens,
        "temporal_mass_gate_threshold": 0.0,
        "temporal_mass_gate_gqa_union": False,
    }


def countcap_direct_budget(history_tokens: int) -> tuple[int, float]:
    """Return the frozen direct-attention token cap and realized fraction."""
    if history_tokens <= 0:
        raise ValueError("history_tokens must be positive")
    target_tokens = max(
        COUNTCAP_DIRECT_MIN_TOKENS,
        math.ceil(COUNTCAP_DIRECT_FRACTION * history_tokens),
    )
    attention_tokens = min(
        history_tokens,
        COUNTCAP_DIRECT_MAX_TOKENS,
        target_tokens,
    )
    return attention_tokens, attention_tokens / history_tokens


def tail_resolution_sample_count(
    target_tail_count: int,
    target_fraction: float,
    *,
    alignment: int = 256,
    minimum: int = 256,
    maximum: int = 8192,
) -> int:
    """Return an aligned sample count with enough expected tail anchors."""
    if target_tail_count <= 0:
        raise ValueError("target_tail_count must be positive")
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be in (0, 1]")
    if alignment <= 0 or minimum <= 0 or maximum < minimum:
        raise ValueError("invalid sample-count bounds")
    raw_sample_count = math.ceil(target_tail_count / target_fraction)
    aligned_sample_count = alignment * math.ceil(
        raw_sample_count / alignment
    )
    return min(maximum, max(minimum, aligned_sample_count))


def sparse_method_config(
    method: str,
    history_tokens: int,
    budget_fractions: tuple[float, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if method in {
        "countcap",
        "countcap_fullprompt",
        "countcap_fullprompt_keypca",
        "countcap_fullprompt_keypca_direct",
        "countcap_fullprompt_keypca_direct_fused",
        COUNTCAP_QK_BALANCED_PACKED_METHOD,
        QKSIEVE_FULLTOPK_METHOD,
        QKSIEVE_MEANVALUE_FULLTOPK_METHOD,
        QKSIEVE_QFUSED_FULLTOPK_METHOD,
        QKSIEVE_FULLTOPK_FP16_METHOD,
        QKSIEVE_FIXED410_FULLTOPK_METHOD,
        QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_METHOD,
        FIER_RTN1_G32_FULLTOPK_METHOD,
        FIER_RTN1_G32_PACKED_FULLTOPK_METHOD,
        QUEST_P16_FULLTOPK_METHOD,
        UNIQUE_P8_FULLTOPK_METHOD,
        RABITQCACHE_RTN1_FULLTOPK_METHOD,
        BINARYPC_OFFLINE64_FULLTOPK_METHOD,
        SPARQ_R32_SELECTOR_FULLTOPK_METHOD,
        SPARQ_R32_FORMULA_FULLTOPK_METHOD,
        QKSIEVE_KEYPCA_UNIFORM1_FULLTOPK_METHOD,
        QKSIEVE_QKBALANCED_UNIFORM1_FULLTOPK_METHOD,
        QKSIEVE_RANDOM_UNIFORM1_FULLTOPK_METHOD,
        QKSIEVE_KEYPCA_AUTOKEY_FULLTOPK_METHOD,
        QKSIEVE_QKBALANCED_AUTOKEY_FULLTOPK_METHOD,
        QKSIEVE_GLOBAL_WMMA_SAMPLED_METHOD,
        QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLED_METHOD,
        QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLEMASS_METHOD,
        QKSIEVE_GLOBAL_KEYMSE_WMMA_PROXYMASS_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_SAMPLED_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_PROXYMASS_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_MEANTAIL_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH8_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH16_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH32_METHOD,
        QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH_PROGRESSIVE_METHOD,
        QKSIEVE_FROZEN_C64_METHOD,
        COUNTCAP_QK_BALANCED_FIXED4421_PACKED_METHOD,
        COUNTCAP_QK_BALANCED_QSCALE_PACKED_METHOD,
        COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_PACKED_METHOD,
        COUNTCAP_QK_BALANCED_QSCALE_OAS_PACKED_METHOD,
        COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_OAS_PACKED_METHOD,
        COUNTCAP_QK_BALANCED_CENTERED_PACKED_METHOD,
        COUNTCAP_QK_BALANCED_SHAREDTAIL_PACKED_METHOD,
        "countcap_fullprompt_keypca_scanqk_fusedav_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused",
        "countcap_fullprompt_keypca_direct_qkvfused_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_asyncprefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_dp4a_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_qproj_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_inplacecache_prefillindex",
        *COUNTCAP_QPROJSCAN_SPLIT_METHODS,
        *COUNTCAP_TEMPORAL_MASS_GATE_METHODS,
        "countcap_fullprompt_keypca_direct_qkvsplit4_prefillindex",
        "countcap_fullprompt_keypca_direct_proxyav_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_reuse2_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_reuse4_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_reuse8_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_gqaunion_reuse2_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvfused_stable16_reuse2_prefillindex",
        "countcap_fullprompt_keypca_direct_qkvwarp",
    }:
        if method == COUNTCAP_QK_BALANCED_PACKED_METHOD:
            score_mode = COUNTCAP_QK_BALANCED_PACKED_SCORE_MODE
        elif method == QKSIEVE_FULLTOPK_METHOD:
            score_mode = QKSIEVE_FULLTOPK_SCORE_MODE
        elif method == QKSIEVE_MEANVALUE_FULLTOPK_METHOD:
            score_mode = QKSIEVE_MEANVALUE_FULLTOPK_SCORE_MODE
        elif method == QKSIEVE_QFUSED_FULLTOPK_METHOD:
            score_mode = QKSIEVE_QFUSED_FULLTOPK_SCORE_MODE
        elif method == QKSIEVE_FULLTOPK_FP16_METHOD:
            score_mode = QKSIEVE_FULLTOPK_FP16_SCORE_MODE
        elif method == QKSIEVE_FIXED410_FULLTOPK_METHOD:
            score_mode = QKSIEVE_FIXED410_FULLTOPK_SCORE_MODE
        elif (
            method
            == QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_METHOD
        ):
            score_mode = (
                QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_SCORE_MODE
            )
        elif method == FIER_RTN1_G32_FULLTOPK_METHOD:
            score_mode = FIER_RTN1_G32_FULLTOPK_SCORE_MODE
        elif method == FIER_RTN1_G32_PACKED_FULLTOPK_METHOD:
            score_mode = FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE
        elif method == QUEST_P16_FULLTOPK_METHOD:
            score_mode = QUEST_P16_FULLTOPK_SCORE_MODE
        elif method == UNIQUE_P8_FULLTOPK_METHOD:
            score_mode = UNIQUE_P8_FULLTOPK_SCORE_MODE
        elif method == RABITQCACHE_RTN1_FULLTOPK_METHOD:
            score_mode = RABITQCACHE_RTN1_FULLTOPK_SCORE_MODE
        elif method == BINARYPC_OFFLINE64_FULLTOPK_METHOD:
            score_mode = BINARYPC_OFFLINE64_FULLTOPK_SCORE_MODE
        elif method == SPARQ_R32_SELECTOR_FULLTOPK_METHOD:
            score_mode = SPARQ_R32_SELECTOR_FULLTOPK_SCORE_MODE
        elif method == SPARQ_R32_FORMULA_FULLTOPK_METHOD:
            score_mode = SPARQ_R32_FORMULA_FULLTOPK_SCORE_MODE
        elif method == QKSIEVE_KEYPCA_UNIFORM1_FULLTOPK_METHOD:
            score_mode = QKSIEVE_KEYPCA_UNIFORM1_FULLTOPK_SCORE_MODE
        elif method == QKSIEVE_QKBALANCED_UNIFORM1_FULLTOPK_METHOD:
            score_mode = QKSIEVE_QKBALANCED_UNIFORM1_FULLTOPK_SCORE_MODE
        elif method == QKSIEVE_RANDOM_UNIFORM1_FULLTOPK_METHOD:
            score_mode = QKSIEVE_RANDOM_UNIFORM1_FULLTOPK_SCORE_MODE
        elif method == QKSIEVE_KEYPCA_AUTOKEY_FULLTOPK_METHOD:
            score_mode = QKSIEVE_KEYPCA_AUTOKEY_FULLTOPK_SCORE_MODE
        elif method == QKSIEVE_QKBALANCED_AUTOKEY_FULLTOPK_METHOD:
            score_mode = QKSIEVE_QKBALANCED_AUTOKEY_FULLTOPK_SCORE_MODE
        elif method == QKSIEVE_GLOBAL_WMMA_SAMPLED_METHOD:
            score_mode = QKSIEVE_GLOBAL_WMMA_SAMPLED_SCORE_MODE
        elif method == QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLED_METHOD:
            score_mode = QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLED_SCORE_MODE
        elif method == QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLEMASS_METHOD:
            score_mode = QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLEMASS_SCORE_MODE
        elif method == QKSIEVE_GLOBAL_KEYMSE_WMMA_PROXYMASS_METHOD:
            score_mode = QKSIEVE_GLOBAL_KEYMSE_WMMA_PROXYMASS_SCORE_MODE
        elif method == QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_SAMPLED_METHOD:
            score_mode = QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLED_SCORE_MODE
        elif method == QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_PROXYMASS_METHOD:
            score_mode = QKSIEVE_GLOBAL_KEYMSE_WMMA_PROXYMASS_SCORE_MODE
        elif method == QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_MEANTAIL_METHOD:
            score_mode = QKSIEVE_GLOBAL_KEYMSE_WMMA_MEANTAIL_SCORE_MODE
        elif method == QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH8_METHOD:
            score_mode = QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH8_SCORE_MODE
        elif method == QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH16_METHOD:
            score_mode = QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH16_SCORE_MODE
        elif method == QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH32_METHOD:
            score_mode = QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH32_SCORE_MODE
        elif (
            method
            == QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH_PROGRESSIVE_METHOD
        ):
            score_mode = (
                QKSIEVE_GLOBAL_KEYMSE_WMMA_VALUESKETCH_PROGRESSIVE_SCORE_MODE
            )
        elif method == QKSIEVE_FROZEN_C64_METHOD:
            score_mode = QKSIEVE_FROZEN_C64_SCORE_MODE
        elif method == COUNTCAP_QK_BALANCED_FIXED4421_PACKED_METHOD:
            score_mode = COUNTCAP_QK_BALANCED_FIXED4421_PACKED_SCORE_MODE
        elif method == COUNTCAP_QK_BALANCED_QSCALE_PACKED_METHOD:
            score_mode = COUNTCAP_QK_BALANCED_QSCALE_PACKED_SCORE_MODE
        elif method == COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_PACKED_METHOD:
            score_mode = (
                COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_PACKED_SCORE_MODE
            )
        elif method == COUNTCAP_QK_BALANCED_QSCALE_OAS_PACKED_METHOD:
            score_mode = COUNTCAP_QK_BALANCED_QSCALE_OAS_PACKED_SCORE_MODE
        elif (
            method
            == COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_OAS_PACKED_METHOD
        ):
            score_mode = (
                COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_OAS_PACKED_SCORE_MODE
            )
        elif method == COUNTCAP_QK_BALANCED_CENTERED_PACKED_METHOD:
            score_mode = COUNTCAP_QK_BALANCED_CENTERED_PACKED_SCORE_MODE
        elif method == COUNTCAP_QK_BALANCED_SHAREDTAIL_PACKED_METHOD:
            score_mode = COUNTCAP_QK_BALANCED_SHAREDTAIL_PACKED_SCORE_MODE
        elif method == "countcap_fullprompt_keypca_direct_qkvwarp":
            score_mode = COUNTCAP_KEYPCA_DIRECT_QKV_WARP_SCORE_MODE
        elif (
            method
            == "countcap_fullprompt_keypca_scanqk_fusedav_prefillindex"
        ):
            score_mode = COUNTCAP_KEYPCA_SCAN_QK_FUSED_AV_SCORE_MODE
        elif (
            method
            == "countcap_fullprompt_keypca_direct_qkvsplit4_prefillindex"
        ):
            score_mode = COUNTCAP_KEYPCA_DIRECT_QKV_SPLIT4_SCORE_MODE
        elif (
            method
            == "countcap_fullprompt_keypca_direct_proxyav_prefillindex"
        ):
            score_mode = COUNTCAP_KEYPCA_DIRECT_PROXY_AV_SCORE_MODE
        elif (
            method
            == "countcap_fullprompt_keypca_direct_qkvfused_gqaunion_reuse2_prefillindex"
        ):
            score_mode = (
                COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_GQA_UNION_REUSE2_SCORE_MODE
            )
        elif (
            method
            == "countcap_fullprompt_keypca_direct_qkvfused_stable16_reuse2_prefillindex"
        ):
            score_mode = (
                COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_STABLE16_REUSE2_SCORE_MODE
            )
        elif method in COUNTCAP_QPROJSCAN_SPLIT_METHODS:
            score_mode = COUNTCAP_QPROJSCAN_SPLIT_METHODS[method]
        elif "_reuse" in method:
            interval = int(method.split("_reuse", 1)[1].split("_", 1)[0])
            score_mode = COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_REUSE_SCORE_MODES[
                interval
            ]
        elif (
            method
            == "countcap_fullprompt_keypca_direct_qkvfused_dp4a_prefillindex"
        ):
            score_mode = COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_DP4A_SCORE_MODE
        elif (
            method
            == "countcap_fullprompt_keypca_direct_qkvfused_qproj_prefillindex"
        ):
            score_mode = COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_QPROJ_SCORE_MODE
        elif (
            method
            in {
                "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_prefillindex",
                "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_inplacecache_prefillindex",
            }
        ):
            score_mode = COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_QPROJSCAN_SCORE_MODE
        elif method in COUNTCAP_TEMPORAL_MASS_GATE_METHODS:
            score_mode = COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_QPROJSCAN_SCORE_MODE
        elif method in {
            "countcap_fullprompt_keypca_direct_qkvfused",
            "countcap_fullprompt_keypca_direct_qkvfused_prefillindex",
            "countcap_fullprompt_keypca_direct_qkvfused_asyncprefillindex",
        }:
            score_mode = COUNTCAP_KEYPCA_DIRECT_QKV_FUSED_SCORE_MODE
        elif method == "countcap_fullprompt_keypca_direct_fused":
            score_mode = COUNTCAP_KEYPCA_DIRECT_FUSED_SCORE_MODE
        elif method == "countcap_fullprompt_keypca_direct":
            score_mode = COUNTCAP_KEYPCA_DIRECT_SCORE_MODE
        elif method == "countcap_fullprompt_keypca":
            score_mode = COUNTCAP_KEYPCA_SCORE_MODE
        else:
            score_mode = COUNTCAP_SCORE_MODE
        config = countcap_config(history_tokens, score_mode=score_mode)
        config["temporal_mass_gate_threshold"] = (
            COUNTCAP_TEMPORAL_MASS_GATE_METHODS.get(method, 0.0)
        )
        config["temporal_mass_gate_gqa_union"] = (
            method in COUNTCAP_TEMPORAL_GQA_MASS_GATE_THRESHOLDS
        )
        if method in QKSIEVE_WMMA_SAMPLED_METHODS:
            config["projection_dim"] = 128
        direct_fraction_override = float(
            getattr(args, "countcap_direct_fraction_override", 0.0)
        )
        if direct_fraction_override > 0.0:
            if not 0.0 < direct_fraction_override < 1.0:
                raise ValueError(
                    "CountCap direct fraction override must be in (0, 1)"
                )
            config["candidate_fraction"] = direct_fraction_override
        if method in {
            "countcap_fullprompt_keypca_direct",
            "countcap_fullprompt_keypca_direct_fused",
            COUNTCAP_QK_BALANCED_PACKED_METHOD,
            QKSIEVE_FULLTOPK_METHOD,
            QKSIEVE_MEANVALUE_FULLTOPK_METHOD,
            QKSIEVE_QFUSED_FULLTOPK_METHOD,
            QKSIEVE_FULLTOPK_FP16_METHOD,
            QKSIEVE_FIXED410_FULLTOPK_METHOD,
            QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_METHOD,
            FIER_RTN1_G32_FULLTOPK_METHOD,
            FIER_RTN1_G32_PACKED_FULLTOPK_METHOD,
            QUEST_P16_FULLTOPK_METHOD,
            UNIQUE_P8_FULLTOPK_METHOD,
            RABITQCACHE_RTN1_FULLTOPK_METHOD,
            BINARYPC_OFFLINE64_FULLTOPK_METHOD,
            SPARQ_R32_SELECTOR_FULLTOPK_METHOD,
            SPARQ_R32_FORMULA_FULLTOPK_METHOD,
            QKSIEVE_KEYPCA_UNIFORM1_FULLTOPK_METHOD,
            QKSIEVE_QKBALANCED_UNIFORM1_FULLTOPK_METHOD,
            QKSIEVE_RANDOM_UNIFORM1_FULLTOPK_METHOD,
            QKSIEVE_KEYPCA_AUTOKEY_FULLTOPK_METHOD,
            QKSIEVE_QKBALANCED_AUTOKEY_FULLTOPK_METHOD,
            QKSIEVE_GLOBAL_WMMA_SAMPLED_METHOD,
            QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLED_METHOD,
            QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLEMASS_METHOD,
            QKSIEVE_GLOBAL_KEYMSE_WMMA_PROXYMASS_METHOD,
            QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_SAMPLED_METHOD,
            QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_PROXYMASS_METHOD,
            QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_MEANTAIL_METHOD,
            QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH8_METHOD,
            QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH16_METHOD,
            QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH32_METHOD,
            QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH_PROGRESSIVE_METHOD,
            QKSIEVE_FROZEN_C64_METHOD,
            COUNTCAP_QK_BALANCED_FIXED4421_PACKED_METHOD,
            COUNTCAP_QK_BALANCED_QSCALE_PACKED_METHOD,
            COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_PACKED_METHOD,
            COUNTCAP_QK_BALANCED_QSCALE_OAS_PACKED_METHOD,
            COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_OAS_PACKED_METHOD,
            COUNTCAP_QK_BALANCED_CENTERED_PACKED_METHOD,
            COUNTCAP_QK_BALANCED_SHAREDTAIL_PACKED_METHOD,
            "countcap_fullprompt_keypca_scanqk_fusedav_prefillindex",
            "countcap_fullprompt_keypca_direct_qkvfused",
            "countcap_fullprompt_keypca_direct_qkvfused_prefillindex",
            "countcap_fullprompt_keypca_direct_qkvfused_asyncprefillindex",
            "countcap_fullprompt_keypca_direct_qkvfused_dp4a_prefillindex",
            "countcap_fullprompt_keypca_direct_qkvfused_qproj_prefillindex",
            "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_prefillindex",
            "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_inplacecache_prefillindex",
            *COUNTCAP_QPROJSCAN_SPLIT_METHODS,
            *COUNTCAP_TEMPORAL_MASS_GATE_METHODS,
            "countcap_fullprompt_keypca_direct_qkvsplit4_prefillindex",
            "countcap_fullprompt_keypca_direct_proxyav_prefillindex",
            "countcap_fullprompt_keypca_direct_qkvfused_reuse2_prefillindex",
            "countcap_fullprompt_keypca_direct_qkvfused_reuse4_prefillindex",
            "countcap_fullprompt_keypca_direct_qkvfused_reuse8_prefillindex",
            "countcap_fullprompt_keypca_direct_qkvfused_gqaunion_reuse2_prefillindex",
            "countcap_fullprompt_keypca_direct_qkvfused_stable16_reuse2_prefillindex",
            "countcap_fullprompt_keypca_direct_qkvwarp",
        }:
            if direct_fraction_override > 0.0:
                direct_fraction = direct_fraction_override
                direct_tokens = max(
                    1,
                    min(
                        history_tokens,
                        round(direct_fraction * history_tokens),
                    ),
                )
            else:
                direct_tokens, direct_fraction = countcap_direct_budget(
                    history_tokens
                )
            config.update(
                budget_fractions=(direct_fraction,),
                candidate_fraction=direct_fraction,
                attention_tokens=direct_tokens,
            )
        if method in QKSIEVE_WMMA_SAMPLED_METHODS:
            config["sampled_quantile_sample_count"] = int(
                getattr(args, "sampled_quantile_sample_count", 256)
            )
        target_tail_count = int(
            getattr(args, "sampled_quantile_target_tail_count", 0)
        )
        if (
            method in QKSIEVE_WMMA_SAMPLED_METHODS
            and target_tail_count > 0
        ):
            target_fraction = float(config["budget_fractions"][0])
            config["sampled_quantile_sample_count"] = (
                tail_resolution_sample_count(
                    target_tail_count,
                    target_fraction,
                )
            )
        if method == QKSIEVE_FROZEN_C64_METHOD:
            target_fraction = float(config["budget_fractions"][0])
            config["sampled_quantile_sample_count"] = (
                tail_resolution_sample_count(64, target_fraction)
            )
        return config
    if method == "exact_top2_fullprompt":
        config = countcap_config(history_tokens, score_mode="exact_qk_top2")
        direct_fraction_override = float(
            getattr(args, "countcap_direct_fraction_override", 0.0)
        )
        if direct_fraction_override > 0.0:
            if not 0.0 < direct_fraction_override <= 1.0:
                raise ValueError(
                    "exact-QK fraction override must be in (0, 1]"
                )
            attention_tokens = min(
                history_tokens,
                max(1, round(direct_fraction_override * history_tokens)),
            )
            attention_fraction = attention_tokens / history_tokens
            config.update(
                budget_fractions=(attention_fraction,),
                attention_tokens=attention_tokens,
            )
        config.update(candidate_fraction=1.0, projection_dim=0)
        return config
    if method in {
        "exact_massadaptive_fullprompt",
        "countcap_massadaptive_fullprompt",
    }:
        fractions = tuple(
            fraction
            for fraction in COUNTCAP_ADAPTIVE_BUDGET_FRACTIONS
            if max(1, round(fraction * history_tokens)) <= COUNTCAP_MAX_ATTENTION_TOKENS
        )
        if not fractions:
            fractions = (COUNTCAP_MAX_ATTENTION_TOKENS / history_tokens,)
        return {
            "budget_fractions": fractions,
            "candidate_fraction": max(0.06, fractions[-1]),
            "projection_dim": (
                0
                if method == "exact_massadaptive_fullprompt"
                else COUNTCAP_PROJECTION_DIM
            ),
            "score_mode": (
                "exact_qk_massadaptive"
                if method == "exact_massadaptive_fullprompt"
                else COUNTCAP_ADAPTIVE_SCORE_MODE
            ),
            "attention_tokens": round(fractions[-1] * history_tokens),
        }
    return {
        "budget_fractions": budget_fractions,
        "candidate_fraction": float(args.candidate_fraction),
        "projection_dim": int(args.projection_dim),
        "score_mode": {
            "qgate_partition": "pca_int4_partition_global_qgate088",
            "ec_bandef": "pca_int4_partition_global_delta16ec95",
            "ec_bandef_budget": "pca_int4_partition_global_delta16ec95_budget",
            "oneshot_bandef_budget": (
                "pca_int4_partition_global_delta16oneshot95_budget"
            ),
        }.get(method, "pca_int4_partition_global_ucb"),
        "attention_tokens": None,
    }


def resolve_method_plan(
    method: str,
    history_tokens: int,
    expected_generated_tokens: int,
    budget_fractions: tuple[float, ...],
    args: argparse.Namespace,
    cost_profile: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    if method != "countcap_auto":
        return (
            method,
            sparse_method_config(method, history_tokens, budget_fractions, args),
            None,
        )
    if cost_profile is None:
        raise ValueError("countcap_auto requires --cost_profile")
    decision = choose_countcap_path(
        cost_profile,
        history_tokens=history_tokens,
        expected_generated_tokens=expected_generated_tokens,
        quality_floor=float(args.cost_quality_floor),
        speed_margin=float(args.cost_speed_margin),
    )
    executed_path = str(decision["selected_path"])
    if executed_path == "full_kv":
        config = {
            "budget_fractions": (1.0,),
            "candidate_fraction": 1.0,
            "projection_dim": 0,
            "score_mode": "full_kv",
            "attention_tokens": history_tokens,
        }
    else:
        config = sparse_method_config(
            executed_path, history_tokens, budget_fractions, args
        )
    return executed_path, config, decision


def parse_tasks(spec: str) -> list[str]:
    tasks = parse_csv_values(spec)
    unknown = [task for task in tasks if task not in lb.LONG_BENCH_PROMPTS]
    if unknown:
        raise ValueError(f"unsupported LongBench tasks: {unknown}")
    if not tasks:
        raise ValueError("at least one task is required")
    return tasks


def load_examples(args: argparse.Namespace) -> list[lb.Example]:
    examples: list[lb.Example] = []
    sample_offset = int(getattr(args, "sample_offset_per_task", 0))
    if sample_offset < 0:
        raise ValueError("sample_offset_per_task must be non-negative")
    for task in parse_tasks(args.tasks):
        info = lb.LONG_BENCH_PROMPTS[task]
        path = args.longbench_data_dir / f"{task}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = rows[sample_offset:]
        if args.max_samples_per_task > 0:
            rows = rows[: args.max_samples_per_task]
        for row_index, row in enumerate(rows):
            if row_index % args.num_shards != args.shard_index:
                continue
            max_new_tokens = int(info["max_new_tokens"])
            if args.max_new_tokens_override > 0:
                max_new_tokens = min(max_new_tokens, args.max_new_tokens_override)
            examples.append(
                lb.Example(
                    benchmark="longbench",
                    task=task,
                    sample_id=str(row.get("_id", row_index)),
                    context=str(row["context"]),
                    query=str(row["input"]),
                    answers=[str(answer) for answer in row["answers"]],
                    prefix_template=str(info["prefix"]),
                    suffix_template=str(info["suffix"]),
                    metric=str(info["metric"]),
                    max_new_tokens=max_new_tokens,
                    length=int(row.get("length", 0) or 0),
                    all_classes=[str(item) for item in (row.get("all_classes") or [])],
                    no_chat=bool(info.get("no_chat", False)),
                )
            )
    return examples


def build_bundle(
    tokenizer: Any, example: lb.Example, args: argparse.Namespace
) -> lb.PromptBundle:
    if args.max_prompt_tokens > 0:
        if (
            getattr(
                args,
                "prompt_truncation_mode",
                "preserve_suffix",
            )
            == "official_middle"
        ):
            return build_official_middle_bundle(
                tokenizer,
                example,
                args.max_prompt_tokens,
                args.prompt_wrapper,
                getattr(args, "official_query_tail_tokens", 8),
            )
        return build_prompt_limited_bundle(
            tokenizer, example, args.max_prompt_tokens, args.prompt_wrapper
        )
    config = SimpleNamespace(
        max_context_tokens=args.max_context_tokens,
        page_tokens=128,
        force_no_chat_tasks="",
        prompt_wrapper=args.prompt_wrapper,
    )
    bundle, _, _, _, _ = lb.build_bundle(tokenizer, example, config)
    return bundle


def _tokenizer_input_ids(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool,
) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
    )
    values = (
        encoded.input_ids
        if hasattr(encoded, "input_ids")
        else encoded["input_ids"]
    )
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return [int(value) for value in values]


def _tokenizer_chat_input_ids(tokenizer: Any, prompt: str) -> list[int]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError(
            "prompt_wrapper=tokenizer_chat requires tokenizer.apply_chat_template"
        )
    values = apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return [int(value) for value in values]


def official_middle_prompt_ids(
    tokenizer: Any,
    example: lb.Example,
    max_prompt_tokens: int,
    prompt_wrapper: str,
) -> list[int]:
    """Reproduce KVCache-Factory's LongBench prompt tokenization order."""

    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    prompt = (
        example.prefix_template
        + example.context
        + example.suffix_template.format(input=example.query)
    )
    raw_ids = _tokenizer_input_ids(
        tokenizer,
        prompt,
        add_special_tokens=True,
    )
    truncated = len(raw_ids) > max_prompt_tokens
    if truncated:
        half = max_prompt_tokens // 2
        prompt = tokenizer.decode(
            raw_ids[:half],
            skip_special_tokens=True,
        ) + tokenizer.decode(
            raw_ids[-half:],
            skip_special_tokens=True,
        )

    wrapped = not example.no_chat and prompt_wrapper != "none"
    if wrapped and prompt_wrapper == "llama3":
        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + prompt
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return _tokenizer_input_ids(
            tokenizer,
            prompt,
            add_special_tokens=False,
        )
    if wrapped and prompt_wrapper == "qwen3":
        prompt = (
            "<|im_start|>user\n"
            + prompt
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        return _tokenizer_input_ids(
            tokenizer,
            prompt,
            add_special_tokens=False,
        )
    if wrapped and prompt_wrapper == "tokenizer_chat":
        return _tokenizer_chat_input_ids(tokenizer, prompt)
    if truncated:
        return _tokenizer_input_ids(
            tokenizer,
            prompt,
            add_special_tokens=True,
        )
    return raw_ids


def build_official_middle_bundle(
    tokenizer: Any,
    example: lb.Example,
    max_prompt_tokens: int,
    prompt_wrapper: str,
    query_tail_tokens: int,
) -> lb.PromptBundle:
    if query_tail_tokens <= 0:
        raise ValueError("query_tail_tokens must be positive")
    prompt_ids = official_middle_prompt_ids(
        tokenizer,
        example,
        max_prompt_tokens,
        prompt_wrapper,
    )
    if len(prompt_ids) < 2:
        raise RuntimeError("official LongBench prompt has fewer than two tokens")
    suffix_count = min(query_tail_tokens, len(prompt_ids) - 1)
    query_start = len(prompt_ids) - suffix_count
    return lb.PromptBundle(
        input_ids=torch.tensor([prompt_ids], dtype=torch.long),
        prefix_token_count=query_start,
        context_token_start=0,
        query_start=query_start,
        suffix_token_count=suffix_count,
        page_spans={},
    )


def build_prompt_limited_bundle(
    tokenizer: Any,
    example: lb.Example,
    max_prompt_tokens: int,
    prompt_wrapper: str,
) -> lb.PromptBundle:
    """Keep the beginning and end of an over-length LongBench prompt.

    AdaKV follows the official LongBench middle-truncation protocol: the first
    half retains the task prefix and document beginning, while the second half
    retains the document end and complete question/instruction suffix.
    """
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    if prompt_wrapper == "tokenizer_chat" and not example.no_chat:
        raise ValueError(
            "prompt_wrapper=tokenizer_chat is supported only with "
            "prompt_truncation_mode=official_middle"
        )

    prefix_text = example.prefix_template
    suffix_text = example.suffix_template.format(input=example.query)
    wrapped = not example.no_chat and prompt_wrapper != "none"
    if wrapped and prompt_wrapper == "llama3":
        prefix_text = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + prefix_text
        )
        suffix_text += (
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif wrapped and prompt_wrapper == "qwen3":
        prefix_text = "<|im_start|>user\n" + prefix_text
        suffix_text += "<|im_end|>\n<|im_start|>assistant\n"

    prefix_ids = lb.token_ids(tokenizer, prefix_text)
    if not wrapped and tokenizer.bos_token_id is not None:
        prefix_ids = [int(tokenizer.bos_token_id), *prefix_ids]
    context_ids = lb.token_ids(tokenizer, example.context)
    suffix_ids = lb.token_ids(tokenizer, suffix_text)

    overhead = len(prefix_ids) + len(suffix_ids)
    if overhead > max_prompt_tokens:
        raise ValueError(
            f"prompt wrappers require {overhead} tokens, exceeding "
            f"max_prompt_tokens={max_prompt_tokens}"
        )
    if overhead + len(context_ids) > max_prompt_tokens:
        context_budget = max_prompt_tokens - overhead
        first_half = max_prompt_tokens // 2
        desired_front_count = max(
            0,
            first_half - len(prefix_ids),
        )
        front_count = min(context_budget, desired_front_count)
        back_count = context_budget - front_count
        context_ids = (
            context_ids[:front_count]
            + (context_ids[-back_count:] if back_count else [])
        )

    prompt_ids = prefix_ids + context_ids + suffix_ids
    if len(prompt_ids) > max_prompt_tokens:
        raise RuntimeError("prompt middle truncation exceeded its token limit")
    query_start = len(prefix_ids) + len(context_ids)
    return lb.PromptBundle(
        input_ids=torch.tensor([prompt_ids], dtype=torch.long),
        prefix_token_count=len(prefix_ids),
        context_token_start=len(prefix_ids),
        query_start=query_start,
        suffix_token_count=len(suffix_ids),
        page_spans={},
    )


def synchronize_cuda_devices() -> None:
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


def empty_cuda_caches() -> None:
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()


def longbench_stop_token_ids(tokenizer: Any, task: str) -> set[int]:
    """Match the official LongBench generation stop policy."""
    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    vocab = tokenizer.get_vocab()
    for token in ("<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"):
        token_id = vocab.get(token)
        if token_id is not None:
            stop_ids.add(int(token_id))
    if task == "samsum":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids:
            stop_ids.add(int(newline_ids[-1]))
    return stop_ids


@torch.inference_mode()
def generate_full(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    bundle: lb.PromptBundle,
    max_new_tokens: int,
    prefill_chunk_tokens: int,
    eos_token_ids: set[int] | None = None,
) -> dict[str, Any]:
    set_attention_implementation(model, "sdpa")
    with (
        head_top_fraction_mode(None),
        head_adaptive_mass_mode(None),
        head_qabs_sampled_mass_mode(None),
    ):
        prefix_cache, prefill_seconds = lb.prefill_prefix(
            model, bundle, input_device, prefill_chunk_tokens
        )
        prediction, generated_ids, query_seconds, decode_seconds = (
            lb.generate_with_cache(
                model,
                tokenizer,
                bundle,
                prefix_cache,
                max_new_tokens,
                input_device,
                eos_token_ids=eos_token_ids,
            )
        )
    return {
        "prediction": prediction,
        "generated_ids": generated_ids,
        "prefill_seconds": prefill_seconds,
        "query_seconds": query_seconds,
        "decode_seconds": decode_seconds,
        "attention_link_ratio": 1.0,
        "exact_qk_ratio": 1.0,
        "estimated_retained_mass": 1.0,
        "temporal_reuse_rate": 0.0,
        "gpu_kv_storage_ratio": 1.0,
        "scan_dimension_fraction": 1.0,
    }


def aggregate_sparse_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "attention_link_ratio": 0.0,
            "exact_qk_ratio": 0.0,
            "estimated_retained_mass": 0.0,
            "temporal_reuse_rate": 0.0,
            "scan_dimension_fraction": 0.0,
            "packed_qmse_fused_query_prepare_requested": 0.0,
            "packed_qmse_fused_query_prepare_executed": 0.0,
            "packed_qmse_allocation_frozen_before_query": 0.0,
            "packed_qmse_fixed_template_active": 0.0,
            "packed_qmse_sample_count": 0.0,
            "packed_qmse_value_sketch_rank": 0.0,
            "packed_qmse_value_sketch_bits": 0.0,
            "packed_qmse_value_sketch_executed": 0.0,
            "packed_qmse_value_sketch_tail_alpha": 0.0,
            "packed_qmse_debug_value_sketch_disabled": 0.0,
            "public_selector_index_bits_per_token": 0.0,
            "public_selector_dimension_count": 0.0,
            "public_selector_selected_page_count": 0.0,
            "public_selector_is_page_granular": 0.0,
            "public_selector_mean_value_correction": 0.0,
            "public_selector_local_window": 0.0,
            "public_selector_approximate_selected_mass": 0.0,
        }
    selected_links = sum(float(row.get("selected_history_links", 0.0)) for row in rows)
    possible_links = sum(float(row.get("possible_history_links", 0.0)) for row in rows)
    if possible_links > 0.0:
        attention_link_ratio = selected_links / possible_links
    else:
        attention_link_ratio = sum(
            float(row.get("attention_link_ratio", 0.0)) for row in rows
        ) / len(rows)
    output = {
        "attention_link_ratio": attention_link_ratio,
        "exact_qk_ratio": sum(
            float(row.get("progressive_exact_qk_fraction_mean", 0.0)) for row in rows
        )
        / len(rows),
        "estimated_retained_mass": sum(
            float(row.get("retained_mass_mean", 0.0)) for row in rows
        )
        / len(rows),
        "temporal_reuse_rate": sum(
            float(row.get("temporal_reuse_rate_mean", 0.0)) for row in rows
        )
        / len(rows),
        "scan_dimension_fraction": sum(
            float(row.get("transport_scan_dimension_fraction_mean", 0.0))
            for row in rows
        )
        / len(rows),
        "auto_split_count": sum(
            float(row.get("auto_split_count_mean", 0.0)) for row in rows
        )
        / len(rows),
        "selected_history_fraction_mean": sum(
            float(row.get("selected_history_fraction_mean", 0.0))
            for row in rows
        )
        / len(rows),
        "selected_history_fraction_p95": sum(
            float(row.get("selected_history_fraction_p95", 0.0))
            for row in rows
        )
        / len(rows),
        "selected_history_fraction_max": max(
            float(row.get("selected_history_fraction_max", 0.0))
            for row in rows
        ),
        "selected_history_count_mean": sum(
            float(row.get("selected_history_count_mean", 0.0))
            for row in rows
        )
        / len(rows),
        "selected_history_count_p95": sum(
            float(row.get("selected_history_count_p95", 0.0))
            for row in rows
        )
        / len(rows),
        "selected_history_count_max": max(
            float(row.get("selected_history_count_max", 0.0))
            for row in rows
        ),
        "sampled_candidate_overflow_fraction": sum(
            float(
                row.get(
                    "sampled_candidate_overflow_fraction_mean",
                    0.0,
                )
            )
            for row in rows
        )
        / len(rows),
        "sampled_quantile_fallback": sum(
            float(row.get("sampled_quantile_fallback_mean", 0.0))
            for row in rows
        )
        / len(rows),
        "packed_qmse_index_bits_per_token": sum(
            float(row.get("packed_qmse_index_bits_per_token_mean", 0.0))
            for row in rows
        )
        / len(rows),
        "packed_qmse_sample_count": sum(
            float(row.get("packed_qmse_sample_count_mean", 0.0))
            for row in rows
        )
        / len(rows),
        "packed_qmse_value_sketch_rank": sum(
            float(row.get("packed_qmse_value_sketch_rank_mean", 0.0))
            for row in rows
        )
        / len(rows),
        "packed_qmse_value_sketch_bits": sum(
            float(row.get("packed_qmse_value_sketch_bits_mean", 0.0))
            for row in rows
        )
        / len(rows),
        "packed_qmse_value_sketch_executed": sum(
            float(row.get("packed_qmse_value_sketch_executed_mean", 0.0))
            for row in rows
        )
        / len(rows),
        "packed_qmse_value_sketch_tail_alpha": sum(
            float(row.get("packed_qmse_value_sketch_tail_alpha_mean", 0.0))
            for row in rows
        )
        / len(rows),
        "packed_qmse_debug_value_sketch_disabled": sum(
            float(
                row.get(
                    "packed_qmse_debug_value_sketch_disabled_mean",
                    0.0,
                )
            )
            for row in rows
        )
        / len(rows),
        "packed_qmse_fused_query_prepare_requested": sum(
            float(
                row.get(
                    "packed_qmse_fused_query_prepare_requested_mean",
                    0.0,
                )
            )
            for row in rows
        )
        / len(rows),
        "packed_qmse_fused_query_prepare_executed": sum(
            float(
                row.get(
                    "packed_qmse_fused_query_prepare_executed_mean",
                    0.0,
                )
            )
            for row in rows
        )
        / len(rows),
        "packed_qmse_allocation_frozen_before_query": sum(
            float(
                row.get(
                    "packed_qmse_allocation_frozen_before_query_mean",
                    0.0,
                )
            )
            for row in rows
        )
        / len(rows),
        "packed_qmse_fixed_template_active": sum(
            float(
                row.get(
                    "packed_qmse_fixed_template_active_mean",
                    0.0,
                )
            )
            for row in rows
        )
        / len(rows),
        "public_selector_index_bits_per_token": sum(
            float(
                row.get(
                    "public_selector_index_bits_per_token_mean",
                    0.0,
                )
            )
            for row in rows
        )
        / len(rows),
        "public_selector_dimension_count": sum(
            float(
                row.get("public_selector_dimension_count_mean", 0.0)
            )
            for row in rows
        )
        / len(rows),
        "public_selector_selected_page_count": sum(
            float(
                row.get(
                    "public_selector_selected_page_count_mean",
                    0.0,
                )
            )
            for row in rows
        )
        / len(rows),
        "public_selector_is_page_granular": sum(
            float(
                row.get(
                    "public_selector_is_page_granular_mean",
                    0.0,
                )
            )
            for row in rows
        )
        / len(rows),
        "public_selector_mean_value_correction": sum(
            float(
                row.get(
                    "public_selector_mean_value_correction_mean",
                    0.0,
                )
            )
            for row in rows
        )
        / len(rows),
        "public_selector_local_window": sum(
            float(row.get("public_selector_local_window_mean", 0.0))
            for row in rows
        )
        / len(rows),
        "public_selector_approximate_selected_mass": sum(
            float(
                row.get(
                    "public_selector_approximate_selected_mass_mean",
                    0.0,
                )
            )
            for row in rows
        )
        / len(rows),
    }
    for name in TEMPORAL_TRACE_METRICS:
        output[name] = sum(
            float(row.get(f"{name}_mean", 0.0)) for row in rows
        ) / len(rows)
    return output


def materialize_packed_qmse_template(
    model: torch.nn.Module,
    template: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Place a model-level QKSieve template beside each decoder layer."""
    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        nested = getattr(decoder, "model", None)
        layers = getattr(nested, "layers", None)
    if layers is None:
        raise TypeError("model does not expose decoder layers")

    resident: dict[int, dict[str, Any]] = {}
    touched_devices: set[torch.device] = set()
    for layer_index, layer_template in template.items():
        normalized_index = int(layer_index)
        if not 0 <= normalized_index < len(layers):
            raise ValueError(
                f"template layer {normalized_index} is outside the model"
            )
        device = next(layers[normalized_index].parameters()).device
        touched_devices.add(device)
        resident[normalized_index] = {
            name: (
                value.to(device=device, non_blocking=False)
                if isinstance(value, torch.Tensor)
                else value
            )
            for name, value in layer_template.items()
        }
    for device in touched_devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return resident


@torch.inference_mode()
def generate_global_partition(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    bundle: lb.PromptBundle,
    max_new_tokens: int,
    prefill_chunk_tokens: int,
    budget_fractions: tuple[float, ...],
    args: argparse.Namespace,
    score_mode: str,
    candidate_fraction: float | None = None,
    projection_dim: int | None = None,
    dense_suffix: bool = False,
    incremental_prefill_index: bool = False,
    asynchronous_prefill_index: bool = False,
    fixed_pca_basis_templates: (
        dict[int, dict[str, torch.Tensor]] | None
    ) = None,
    fixed_packed_qmse_templates: (
        dict[int, dict[str, Any]] | None
    ) = None,
    fixed_binarypc_projections: dict[int, torch.Tensor] | None = None,
    captured_pca_basis_templates: (
        dict[int, dict[str, torch.Tensor]] | None
    ) = None,
    attention_record_sink: list[dict[str, Any]] | None = None,
    eos_token_ids: set[int] | None = None,
    temporal_mass_gate_threshold: float = 0.0,
    temporal_mass_gate_gqa_union: bool = False,
    use_preallocated_cache: bool = False,
    query_calibration_tokens: int = 8,
    prefill_query_record_sink: dict[int, torch.Tensor] | None = None,
    prefill_query_record_tokens: int | None = None,
    sampled_quantile_sample_count: int | None = None,
) -> dict[str, Any]:
    if asynchronous_prefill_index and not incremental_prefill_index:
        raise ValueError("asynchronous index build requires incremental prefill")
    if query_calibration_tokens <= 0:
        raise ValueError("query_calibration_tokens must be positive")
    if (
        prefill_query_record_tokens is not None
        and prefill_query_record_tokens <= 0
    ):
        raise ValueError("prefill_query_record_tokens must be positive")
    packed_qk_balanced = score_mode in PACKED_QUERY_CALIBRATED_SCORE_MODES
    rabitq_reference = score_mode == RABITQCACHE_RTN1_FULLTOPK_SCORE_MODE
    if packed_qk_balanced and not dense_suffix:
        raise ValueError(
            "QK-balanced packed retrieval requires a dense prompt suffix "
            "for query calibration"
        )
    with head_qabs_sampled_mass_mode(
        mass_threshold=args.mass_threshold,
        budget_fractions=budget_fractions,
        sample_fraction=args.sample_fraction,
        qabs_dim_count=8,
        candidate_fraction=(
            args.candidate_fraction
            if candidate_fraction is None
            else candidate_fraction
        ),
        use_cuda_kernels=True,
        skip_candidate_rerank=False,
        qabs_int2_onthefly=False,
        early_layer_count=0,
        early_budget_fraction=budget_fractions[0],
        score_mode=score_mode,
        projection_dim=(
            args.projection_dim if projection_dim is None else projection_dim
        ),
        gqa_candidate_mode="independent",
        adaptive_rank_energy_threshold=0.85,
        adaptive_rank_residual_precision="int4",
        value_mass_threshold=args.value_mass_threshold,
        partition_ucb_z=args.partition_ucb_z,
        partition_overfetch_factor=args.partition_overfetch_factor,
        temporal_mass_gate_threshold=temporal_mass_gate_threshold,
        temporal_mass_gate_gqa_union=temporal_mass_gate_gqa_union,
        qk_metric_query_shrinkage=float(
            getattr(args, "qk_metric_query_shrinkage", 0.75)
        ),
        sampled_quantile_sample_count=int(
            getattr(args, "sampled_quantile_sample_count", 256)
            if sampled_quantile_sample_count is None
            else sampled_quantile_sample_count
        ),
        direct_min_tokens=(
            COUNTCAP_DIRECT_MIN_TOKENS
            if score_mode in QKSIEVE_GLOBAL_WMMA_SAMPLED_SCORE_MODES
            else 0
        ),
        direct_max_tokens=(
            COUNTCAP_DIRECT_MAX_TOKENS
            if score_mode in QKSIEVE_GLOBAL_WMMA_SAMPLED_SCORE_MODES
            else 0
        ),
    ):
        if fixed_pca_basis_templates is not None:
            install_active_keypca_basis_templates(
                fixed_pca_basis_templates
            )
        if fixed_packed_qmse_templates is not None:
            install_active_packed_qmse_templates(
                fixed_packed_qmse_templates
            )
        if fixed_binarypc_projections is not None:
            install_active_binarypc_projections(
                fixed_binarypc_projections
            )
        model_config = getattr(model, "config", None)
        query_head_count = (
            int(getattr(model_config, "num_attention_heads", 0))
            if incremental_prefill_index
            else 0
        )
        index_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        index_stream = (
            torch.cuda.Stream(device=input_device)
            if asynchronous_prefill_index and torch.cuda.is_available()
            else None
        )

        def update_prefill_index(active_cache: Any) -> None:
            if not incremental_prefill_index:
                return
            if query_head_count <= 0:
                raise ValueError("model config does not define attention heads")
            if index_stream is not None:
                ready = torch.cuda.Event()
                ready.record()
                with torch.cuda.stream(index_stream):
                    index_stream.wait_event(ready)
                    started = torch.cuda.Event(enable_timing=True)
                    finished = torch.cuda.Event(enable_timing=True)
                    started.record()
                    prebuild_active_keypca_int4_index(
                        active_cache,
                        query_head_count,
                    )
                    finished.record()
                index_events.append((started, finished))
            else:
                if torch.cuda.is_available():
                    started = torch.cuda.Event(enable_timing=True)
                    finished = torch.cuda.Event(enable_timing=True)
                    started.record()
                prebuild_active_keypca_int4_index(
                    active_cache,
                    query_head_count,
                )
                if torch.cuda.is_available():
                    finished.record()
                    index_events.append((started, finished))

        set_attention_implementation(model, "sdpa")
        initial_cache = (
            PreallocatedDynamicCache(
                max_cache_len=(
                    int(bundle.input_ids.shape[-1]) + max_new_tokens + 8
                )
            )
            if use_preallocated_cache
            else None
        )
        query_moment_records: dict[int, dict[str, Any]] = {}
        query_moment_context = (
            prefill_query_moment_mode(query_moment_records)
            if rabitq_reference
            else nullcontext(query_moment_records)
        )
        with query_moment_context:
            if incremental_prefill_index:
                prefix_cache, prefill_seconds = lb.prefill_prefix(
                    model,
                    bundle,
                    input_device,
                    prefill_chunk_tokens,
                    cache_chunk_callback=update_prefill_index,
                    initial_cache=initial_cache,
                )
            else:
                with head_qabs_sampled_mass_mode(None):
                    prefix_cache, prefill_seconds = lb.prefill_prefix(
                        model,
                        bundle,
                        input_device,
                        prefill_chunk_tokens,
                        initial_cache=initial_cache,
                    )

        cache = prefix_cache
        previous_logits: torch.Tensor | None = None
        sparse_stats: list[dict[str, float]] = []
        qk_prebuild_stats: dict[str, float | int] = {}
        suffix_tensor = bundle.input_ids[:, bundle.query_start :]
        suffix_ids = suffix_tensor[0].tolist()
        nvtx_profile_active = (
            os.environ.get("COUNTCAP_NVTX_PROFILE", "0") == "1"
            and torch.cuda.is_available()
        )
        cuda_profile_active = (
            os.environ.get("COUNTCAP_CUDA_PROFILE", "0") == "1"
            and torch.cuda.is_available()
        )
        if nvtx_profile_active:
            torch.cuda.nvtx.range_push("countcap_online")
        if cuda_profile_active:
            torch.cuda.cudart().cudaProfilerStart()
        if not suffix_ids:
            raise RuntimeError("LongBench suffix must contain at least one token")

        # The question/instruction suffix is short and already available before
        # generation. Processing it as one dense segment preserves exact prompt
        # conditioning and avoids dozens of one-token sparse-attention launches.
        query_seconds = 0.0
        if dense_suffix:
            set_attention_implementation(model, "sdpa")
            captured_query_tokens = max(
                query_calibration_tokens,
                (
                    int(prefill_query_record_tokens)
                    if prefill_query_record_tokens is not None
                    else query_calibration_tokens
                ),
            )
            query_capture = (
                prefill_query_tail_mode(captured_query_tokens)
                if packed_qk_balanced
                else nullcontext({})
            )
            query_moment_capture = (
                prefill_query_moment_mode(query_moment_records)
                if rabitq_reference
                else nullcontext(query_moment_records)
            )
            with (
                head_qabs_sampled_mass_mode(None),
                query_capture as queries,
                query_moment_capture,
            ):
                cache, previous_logits, query_seconds = lb.run_token_segment(
                    model,
                    suffix_tensor,
                    cache,
                    bundle.query_start,
                    input_device,
                )
            if packed_qk_balanced:
                calibration_queries = {
                    int(layer): query[
                        ..., -query_calibration_tokens:, :
                    ]
                    for layer, query in queries.items()
                }
                seed_packed_qmse_prefill_queries(calibration_queries)
                parallel_qk_workers = int(
                    os.environ.get("QKSIEVE_PARALLEL_QK_WORKERS", "0")
                )
                if (
                    score_mode == QKSIEVE_FROZEN_C64_SCORE_MODE
                    and parallel_qk_workers > 0
                ):
                    qk_prebuild_stats = (
                        precompute_active_packed_qmse_qk_factors(
                            cache,
                            max_workers=parallel_qk_workers,
                        )
                    )
                    query_seconds += float(
                        qk_prebuild_stats.get("total_seconds", 0.0)
                    )
                if prefill_query_record_sink is not None:
                    prefill_query_record_sink.clear()
                    prefill_query_record_sink.update(
                        {
                            int(layer): query.detach().cpu()
                            for layer, query in queries.items()
                        }
                    )
            if rabitq_reference:
                seed_rabitq_prefill_query_moments(query_moment_records)
            if incremental_prefill_index:
                synchronize_cuda_devices()
                query_index_started = time.perf_counter()
                update_prefill_index(cache)
                synchronize_cuda_devices()
                query_seconds += time.perf_counter() - query_index_started
            suffix_ids = []

        set_attention_implementation(model, "eager")
        if suffix_ids:
            synchronize_cuda_devices()
            query_started = time.perf_counter()
            for offset, token_id in enumerate(suffix_ids):
                token_stats_kwargs: dict[str, Any] = {
                    "collect_attention_stats": args.collect_attention_stats,
                }
                if attention_record_sink is not None:
                    token_stats_kwargs["attention_record_sink"] = (
                        attention_record_sink
                    )
                cache, previous_logits, _, features = run_one_token(
                    model,
                    int(token_id),
                    cache,
                    bundle.query_start + offset,
                    input_device,
                    **token_stats_kwargs,
                )
                if features:
                    sparse_stats.append(features)
            synchronize_cuda_devices()
            query_seconds = time.perf_counter() - query_started

        if previous_logits is None:
            raise RuntimeError("LongBench suffix produced no logits")
        eos_ids = set(eos_token_ids or ())
        if not eos_ids and tokenizer.eos_token_id is not None:
            eos_ids.add(int(tokenizer.eos_token_id))
        generated_ids: list[int] = []
        synchronize_cuda_devices()
        decode_started = time.perf_counter()
        for step in range(max_new_tokens):
            next_id = int(torch.argmax(previous_logits.float(), dim=-1).item())
            if next_id in eos_ids:
                break
            generated_ids.append(next_id)
            if step + 1 == max_new_tokens:
                break
            token_stats_kwargs = {
                "collect_attention_stats": args.collect_attention_stats,
            }
            if attention_record_sink is not None:
                token_stats_kwargs["attention_record_sink"] = (
                    attention_record_sink
                )
            cache, previous_logits, _, features = run_one_token(
                model,
                next_id,
                cache,
                int(bundle.input_ids.shape[-1]) + step,
                input_device,
                **token_stats_kwargs,
            )
            if features:
                sparse_stats.append(features)
        synchronize_cuda_devices()
        decode_seconds = time.perf_counter() - decode_started
        if nvtx_profile_active:
            torch.cuda.nvtx.range_pop()
        if cuda_profile_active:
            torch.cuda.cudart().cudaProfilerStop()
        if captured_pca_basis_templates is not None:
            captured_pca_basis_templates.clear()
            captured_pca_basis_templates.update(
                export_active_keypca_basis_templates()
            )

    index_build_seconds = 0.0
    if index_events:
        index_build_seconds = sum(
            started.elapsed_time(finished) / 1000.0
            for started, finished in index_events
        )
    stats = aggregate_sparse_stats(sparse_stats)
    return {
        "prediction": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "generated_ids": generated_ids,
        "prefill_seconds": prefill_seconds,
        "query_seconds": query_seconds,
        "decode_seconds": decode_seconds,
        "index_build_seconds": index_build_seconds,
        "qk_prebuild_seconds": float(
            qk_prebuild_stats.get("total_seconds", 0.0)
        ),
        "qk_prebuild_layers": int(qk_prebuild_stats.get("layers", 0)),
        "qk_batched_allocation_layers": int(
            qk_prebuild_stats.get("batched_allocation_layers", 0)
        ),
        **stats,
        # This quality harness uses a standard DynamicCache. Physical KV offload is
        # deliberately reported separately from sparse attention computation.
        "gpu_kv_storage_ratio": 1.0,
    }


@torch.inference_mode()
def generate_exact_sparse(
    model: torch.nn.Module,
    tokenizer: Any,
    input_device: torch.device,
    bundle: lb.PromptBundle,
    max_new_tokens: int,
    prefill_chunk_tokens: int,
    budget_fractions: tuple[float, ...],
    args: argparse.Namespace,
    *,
    adaptive_mass: bool,
    eos_token_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Quality-only exact-QK control with dense prompt encoding."""
    set_attention_implementation(model, "sdpa")
    with (
        head_top_fraction_mode(None),
        head_adaptive_mass_mode(None),
        head_qabs_sampled_mass_mode(None),
    ):
        prefix_cache, prefill_seconds = lb.prefill_prefix(
            model, bundle, input_device, prefill_chunk_tokens
        )
        cache, previous_logits, query_seconds = lb.run_token_segment(
            model,
            bundle.input_ids[:, bundle.query_start :],
            prefix_cache,
            bundle.query_start,
            input_device,
        )

    sparse_stats: list[dict[str, float]] = []
    set_attention_implementation(model, "eager")
    exact_mode = (
        head_adaptive_mass_mode(
            args.mass_threshold, budget_fractions, sample_fraction=None
        )
        if adaptive_mass
        else head_top_fraction_mode(budget_fractions[-1])
    )
    with head_qabs_sampled_mass_mode(None), exact_mode:
        eos_ids = set(eos_token_ids or ())
        if not eos_ids and tokenizer.eos_token_id is not None:
            eos_ids.add(int(tokenizer.eos_token_id))
        generated_ids: list[int] = []
        synchronize_cuda_devices()
        decode_started = time.perf_counter()
        for step in range(max_new_tokens):
            next_id = int(torch.argmax(previous_logits.float(), dim=-1).item())
            if next_id in eos_ids:
                break
            generated_ids.append(next_id)
            if step + 1 == max_new_tokens:
                break
            cache, previous_logits, _, features = run_one_token(
                model,
                next_id,
                cache,
                int(bundle.input_ids.shape[-1]) + step,
                input_device,
                collect_attention_stats=args.collect_attention_stats,
            )
            if features:
                sparse_stats.append(features)
        synchronize_cuda_devices()
        decode_seconds = time.perf_counter() - decode_started

    stats = aggregate_sparse_stats(sparse_stats)
    if not adaptive_mass:
        stats["attention_link_ratio"] = budget_fractions[-1]
    stats["exact_qk_ratio"] = 1.0
    stats["scan_dimension_fraction"] = 1.0
    return {
        "prediction": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "generated_ids": generated_ids,
        "prefill_seconds": prefill_seconds,
        "query_seconds": query_seconds,
        "decode_seconds": decode_seconds,
        **stats,
        "gpu_kv_storage_ratio": 1.0,
    }


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    fieldnames = list(row)
    if not needs_header:
        with path.open("r", encoding="utf-8", newline="") as existing:
            existing_header = csv.DictReader(existing).fieldnames
        if not existing_header:
            raise RuntimeError(f"existing CSV has no header: {path}")
        fieldnames = existing_header
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    tasks = sorted({str(row["task"]) for row in rows})
    full_online_by_task = {
        task: mean(
            [row for row in rows if row["task"] == task and row["method"] == "full_kv"],
            "online_seconds",
        )
        for task in tasks
        if any(row["task"] == task and row["method"] == "full_kv" for row in rows)
    }
    for method in methods:
        method_task_scores: list[float] = []
        for task in tasks:
            subset = [
                row for row in rows if row["task"] == task and row["method"] == method
            ]
            if not subset:
                continue
            score = mean(subset, "score")
            online_seconds = mean(subset, "online_seconds")
            method_task_scores.append(score)
            output.append(
                {
                    "task": task,
                    "method": method,
                    "samples": len(subset),
                    "score": score,
                    "mean_prompt_tokens": mean(subset, "prompt_tokens"),
                    "mean_generated_tokens": mean(subset, "generated_tokens"),
                    "mean_configured_attention_fraction": mean(
                        subset, "configured_attention_fraction"
                    ),
                    "mean_configured_candidate_fraction": mean(
                        subset, "configured_candidate_fraction"
                    ),
                    "mean_attention_link_ratio": mean(subset, "attention_link_ratio"),
                    "mean_exact_qk_ratio": mean(subset, "exact_qk_ratio"),
                    "mean_temporal_reuse_rate": mean(
                        subset, "temporal_reuse_rate"
                    ),
                    "mean_gpu_kv_storage_ratio": mean(subset, "gpu_kv_storage_ratio"),
                    "mean_scan_dimension_fraction": mean(
                        subset, "scan_dimension_fraction"
                    ),
                    "mean_online_seconds": online_seconds,
                    "paired_online_speedup": (
                        full_online_by_task[task] / online_seconds
                        if task in full_online_by_task
                        else ""
                    ),
                }
            )
        subset = [row for row in rows if row["method"] == method]
        if subset:
            paired_full = [
                row
                for row in rows
                if row["method"] == "full_kv"
                and (row["task"], row["sample_id"])
                in {(item["task"], item["sample_id"]) for item in subset}
            ]
            output.append(
                {
                    "task": "ALL",
                    "method": method,
                    "samples": len(subset),
                    "score": sum(method_task_scores) / len(method_task_scores),
                    "mean_prompt_tokens": mean(subset, "prompt_tokens"),
                    "mean_generated_tokens": mean(subset, "generated_tokens"),
                    "mean_configured_attention_fraction": mean(
                        subset, "configured_attention_fraction"
                    ),
                    "mean_configured_candidate_fraction": mean(
                        subset, "configured_candidate_fraction"
                    ),
                    "mean_attention_link_ratio": mean(subset, "attention_link_ratio"),
                    "mean_exact_qk_ratio": mean(subset, "exact_qk_ratio"),
                    "mean_temporal_reuse_rate": mean(
                        subset, "temporal_reuse_rate"
                    ),
                    "mean_gpu_kv_storage_ratio": mean(subset, "gpu_kv_storage_ratio"),
                    "mean_scan_dimension_fraction": mean(
                        subset, "scan_dimension_fraction"
                    ),
                    "mean_online_seconds": mean(subset, "online_seconds"),
                    "paired_online_speedup": (
                        mean(paired_full, "online_seconds")
                        / mean(subset, "online_seconds")
                        if paired_full
                        else ""
                    ),
                }
            )
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if not 0.0 < args.mass_threshold <= 1.0:
        raise ValueError("mass_threshold must be in (0, 1]")
    if not 0.0 <= args.qk_metric_query_shrinkage <= 1.0:
        raise ValueError("qk_metric_query_shrinkage must be in [0, 1]")
    if not 0.0 < args.sample_fraction <= args.candidate_fraction <= 1.0:
        raise ValueError("expected sample_fraction <= candidate_fraction in (0, 1]")
    methods = parse_methods(args.methods)
    if (
        any(
            method in QKSIEVE_GLOBAL_WMMA_SAMPLED_METHODS
            for method in methods
        )
        and args.packed_qmse_template_in is None
    ):
        raise ValueError(
            f"{QKSIEVE_GLOBAL_WMMA_SAMPLED_METHOD} requires "
            "--packed_qmse_template_in"
        )
    if (
        BINARYPC_OFFLINE64_FULLTOPK_METHOD in methods
        and args.binarypc_projection_path is None
    ):
        raise ValueError(
            f"{BINARYPC_OFFLINE64_FULLTOPK_METHOD} requires "
            "--binarypc_projection_path"
        )
    valid_sample_count = (
        args.sampled_quantile_sample_count == 128
        or (
            256 <= args.sampled_quantile_sample_count <= 1024
            and args.sampled_quantile_sample_count % 256 == 0
        )
    )
    if not valid_sample_count:
        raise ValueError(
            "sampled_quantile_sample_count must be 128 or a multiple of "
            "256 in [256, 1024]"
        )
    if not 0 <= args.sampled_quantile_target_tail_count <= 8192:
        raise ValueError(
            "sampled_quantile_target_tail_count must be in [0, 8192]"
        )
    qk_trace_config = resolve_qk_trace_config(args, methods)
    if qk_trace_config is not None:
        qk_trace_config["output_dir"].mkdir(parents=True, exist_ok=True)
    if args.cost_expected_generation_tokens < 0:
        raise ValueError("cost_expected_generation_tokens must be non-negative")
    if args.max_prompt_tokens < 0:
        raise ValueError("max_prompt_tokens must be non-negative")
    if args.official_query_tail_tokens <= 0:
        raise ValueError("official_query_tail_tokens must be positive")
    cost_profile = (
        load_cost_profile(args.cost_profile) if args.cost_profile is not None else None
    )
    if "countcap_auto" in methods and cost_profile is None:
        raise ValueError("countcap_auto requires --cost_profile")
    budgets = parse_budget_fractions(args.budget_fractions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(args)
    preload_extensions = (
        os.environ.get("QKSIEVE_PRELOAD_EXTENSIONS", "")
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )
    if preload_extensions:
        preload_qksieve_runtime_extensions()
        if (
            os.environ.get("QKSIEVE_PRELOAD_QMSE_RATE_TABLES", "1")
            .strip()
            .lower()
            not in {"0", "false", "no"}
        ):
            preload_qksieve_qmse_rate_tables(model)
        precompute_qksieve_value_metric_grams(model)
    fixed_packed_qmse_templates = None
    if any(
        method in QKSIEVE_GLOBAL_WMMA_SAMPLED_METHODS
        for method in methods
    ):
        assert args.packed_qmse_template_in is not None
        loaded_template = torch.load(
            args.packed_qmse_template_in,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(loaded_template, dict):
            raise TypeError(
                "packed qMSE template must be a layer dictionary"
            )
        fixed_packed_qmse_templates = materialize_packed_qmse_template(
            model,
            {int(layer): value for layer, value in loaded_template.items()},
        )
    fixed_binarypc_projections = None
    if BINARYPC_OFFLINE64_FULLTOPK_METHOD in methods:
        assert args.binarypc_projection_path is not None
        loaded_projections = torch.load(
            args.binarypc_projection_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(loaded_projections, dict):
            raise TypeError("BinaryPC projection checkpoint must be a dictionary")
        fixed_binarypc_projections = {
            int(layer): projection.detach().cpu()
            for layer, projection in loaded_projections.items()
            if isinstance(projection, torch.Tensor)
        }
        expected_layers = int(getattr(model.config, "num_hidden_layers", 0))
        if set(fixed_binarypc_projections) != set(range(expected_layers)):
            raise ValueError(
                "BinaryPC projection checkpoint does not cover every model layer"
            )
    examples = load_examples(args)
    results_path = args.output_dir / "sample_results.csv"
    rows = read_csv(results_path)
    completed = {
        (str(row["task"]), str(row["sample_id"]), str(row["method"]))
        for row in rows
    }
    for index, example in enumerate(examples, start=1):
        bundle = build_bundle(tokenizer, example, args)
        eos_token_ids = longbench_stop_token_ids(tokenizer, example.task)
        print(
            f"[{index}/{len(examples)}] {example.task}/{example.sample_id} "
            f"prefix={bundle.query_start} suffix={bundle.suffix_token_count}",
            flush=True,
        )
        active_methods = list(methods)
        if bundle.query_start < args.minimum_sparse_prefix_tokens:
            active_methods = [
                method
                for method in active_methods
                if method
                not in {
                    "global_partition",
                    "qgate_partition",
                    "ec_bandef",
                    "ec_bandef_budget",
                    "oneshot_bandef_budget",
                    "countcap",
                    "countcap_fullprompt",
                    "countcap_fullprompt_keypca",
                    "countcap_fullprompt_keypca_direct",
                    "countcap_fullprompt_keypca_direct_fused",
                    COUNTCAP_QK_BALANCED_PACKED_METHOD,
                    QKSIEVE_FULLTOPK_METHOD,
                    QKSIEVE_MEANVALUE_FULLTOPK_METHOD,
                    QKSIEVE_QFUSED_FULLTOPK_METHOD,
                    QKSIEVE_FULLTOPK_FP16_METHOD,
                    QKSIEVE_FIXED410_FULLTOPK_METHOD,
                    QKSIEVE_FIXED410_PRERERANK_L00TO08_FULLTOPK_METHOD,
                    QKSIEVE_GLOBAL_WMMA_SAMPLED_METHOD,
                    QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLED_METHOD,
                    QKSIEVE_GLOBAL_KEYMSE_WMMA_SAMPLEMASS_METHOD,
                    QKSIEVE_GLOBAL_KEYMSE_WMMA_PROXYMASS_METHOD,
                    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_SAMPLED_METHOD,
                    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_PROXYMASS_METHOD,
                    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_MEANTAIL_METHOD,
                    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH8_METHOD,
                    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH16_METHOD,
                    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH32_METHOD,
                    QKSIEVE_REQUESTLOCAL_KEYMSE_WMMA_VALUESKETCH_PROGRESSIVE_METHOD,
                    QKSIEVE_FROZEN_C64_METHOD,
                    QUEST_P16_FULLTOPK_METHOD,
                    UNIQUE_P8_FULLTOPK_METHOD,
                    RABITQCACHE_RTN1_FULLTOPK_METHOD,
                    BINARYPC_OFFLINE64_FULLTOPK_METHOD,
                    SPARQ_R32_SELECTOR_FULLTOPK_METHOD,
                    SPARQ_R32_FORMULA_FULLTOPK_METHOD,
                    COUNTCAP_QK_BALANCED_FIXED4421_PACKED_METHOD,
                    COUNTCAP_QK_BALANCED_QSCALE_PACKED_METHOD,
                    COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_PACKED_METHOD,
                    COUNTCAP_QK_BALANCED_QSCALE_OAS_PACKED_METHOD,
                    COUNTCAP_QK_BALANCED_FIXED4421_QSCALE_OAS_PACKED_METHOD,
                    COUNTCAP_QK_BALANCED_CENTERED_PACKED_METHOD,
                    COUNTCAP_QK_BALANCED_SHAREDTAIL_PACKED_METHOD,
                    "countcap_fullprompt_keypca_scanqk_fusedav_prefillindex",
                    "countcap_fullprompt_keypca_direct_qkvfused",
                    "countcap_fullprompt_keypca_direct_qkvfused_prefillindex",
                    "countcap_fullprompt_keypca_direct_qkvfused_asyncprefillindex",
                    "countcap_fullprompt_keypca_direct_qkvfused_dp4a_prefillindex",
                    "countcap_fullprompt_keypca_direct_qkvfused_qproj_prefillindex",
                    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_prefillindex",
                    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_inplacecache_prefillindex",
                    *COUNTCAP_QPROJSCAN_SPLIT_METHODS,
                    *COUNTCAP_TEMPORAL_MASS_GATE_METHODS,
                    "countcap_fullprompt_keypca_direct_qkvsplit4_prefillindex",
                    "countcap_fullprompt_keypca_direct_proxyav_prefillindex",
                    "countcap_fullprompt_keypca_direct_qkvfused_reuse2_prefillindex",
                    "countcap_fullprompt_keypca_direct_qkvfused_reuse4_prefillindex",
                    "countcap_fullprompt_keypca_direct_qkvfused_reuse8_prefillindex",
                    "countcap_fullprompt_keypca_direct_qkvfused_gqaunion_reuse2_prefillindex",
                    "countcap_fullprompt_keypca_direct_qkvfused_stable16_reuse2_prefillindex",
                    "countcap_fullprompt_keypca_direct_qkvwarp",
                    "exact_top2_fullprompt",
                    "exact_massadaptive_fullprompt",
                    "countcap_massadaptive_fullprompt",
                }
            ]
        for method in active_methods:
            key = (example.task, example.sample_id, method)
            if key in completed:
                print(f"  {method}: already complete", flush=True)
                continue
            gate_decision: dict[str, Any] | None = None
            executed_path = method
            if method == "full_kv":
                method_config = {
                    "budget_fractions": (1.0,),
                    "candidate_fraction": 1.0,
                    "projection_dim": 0,
                    "score_mode": "full_kv",
                    "attention_tokens": bundle.query_start,
                }
                result = generate_full(
                    model,
                    tokenizer,
                    input_device,
                    bundle,
                    example.max_new_tokens,
                    args.prefill_chunk_tokens,
                    eos_token_ids,
                )
            else:
                expected_generated_tokens = (
                    args.cost_expected_generation_tokens
                    if args.cost_expected_generation_tokens > 0
                    else example.max_new_tokens
                )
                executed_path, method_config, gate_decision = resolve_method_plan(
                    method,
                    bundle.query_start,
                    expected_generated_tokens,
                    budgets,
                    args,
                    cost_profile,
                )
                if executed_path == "full_kv":
                    result = generate_full(
                        model,
                        tokenizer,
                        input_device,
                        bundle,
                        example.max_new_tokens,
                        args.prefill_chunk_tokens,
                        eos_token_ids,
                    )
                elif executed_path in {
                    "exact_top2_fullprompt",
                    "exact_massadaptive_fullprompt",
                }:
                    result = generate_exact_sparse(
                        model,
                        tokenizer,
                        input_device,
                        bundle,
                        example.max_new_tokens,
                        args.prefill_chunk_tokens,
                        method_config["budget_fractions"],
                        args,
                        adaptive_mass=(
                            executed_path == "exact_massadaptive_fullprompt"
                        ),
                        eos_token_ids=eos_token_ids,
                    )
                else:
                    trace_enabled = bool(
                        qk_trace_config is not None
                        and method == qk_trace_config["method"]
                        and executed_path == method
                    )
                    trace_records: list[dict[str, Any]] = []
                    prefill_query_records: dict[int, torch.Tensor] = {}
                    trace_context = (
                        capture_qk_trace(
                            trace_records,
                            layers=qk_trace_config["layers"],
                            max_records_per_layer=len(
                                qk_trace_config["steps"]
                            ),
                            state_on_first_record_only=True,
                            record_steps=qk_trace_config["steps"],
                        )
                        if trace_enabled
                        else nullcontext()
                    )
                    with trace_context:
                        result = generate_global_partition(
                            model,
                            tokenizer,
                            input_device,
                            bundle,
                            example.max_new_tokens,
                            args.prefill_chunk_tokens,
                            method_config["budget_fractions"],
                            args,
                            method_config["score_mode"],
                            candidate_fraction=method_config["candidate_fraction"],
                            projection_dim=method_config["projection_dim"],
                            dense_suffix=uses_dense_prompt_suffix(executed_path),
                            incremental_prefill_index=(
                                executed_path
                                in {
                                    "countcap_fullprompt_keypca_direct_qkvfused_prefillindex",
                                    "countcap_fullprompt_keypca_scanqk_fusedav_prefillindex",
                                    "countcap_fullprompt_keypca_direct_qkvfused_asyncprefillindex",
                                    "countcap_fullprompt_keypca_direct_qkvfused_dp4a_prefillindex",
                                    "countcap_fullprompt_keypca_direct_qkvfused_qproj_prefillindex",
                                    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_prefillindex",
                                    "countcap_fullprompt_keypca_direct_qkvfused_qprojscan_inplacecache_prefillindex",
                                    *COUNTCAP_QPROJSCAN_SPLIT_METHODS,
                                    *COUNTCAP_TEMPORAL_MASS_GATE_METHODS,
                                    "countcap_fullprompt_keypca_direct_qkvsplit4_prefillindex",
                                    "countcap_fullprompt_keypca_direct_proxyav_prefillindex",
                                    "countcap_fullprompt_keypca_direct_qkvfused_reuse2_prefillindex",
                                    "countcap_fullprompt_keypca_direct_qkvfused_reuse4_prefillindex",
                                    "countcap_fullprompt_keypca_direct_qkvfused_reuse8_prefillindex",
                                    "countcap_fullprompt_keypca_direct_qkvfused_gqaunion_reuse2_prefillindex",
                                    "countcap_fullprompt_keypca_direct_qkvfused_stable16_reuse2_prefillindex",
                                }
                            ),
                            asynchronous_prefill_index=(
                                executed_path
                                == "countcap_fullprompt_keypca_direct_qkvfused_asyncprefillindex"
                            ),
                            eos_token_ids=eos_token_ids,
                            temporal_mass_gate_threshold=method_config.get(
                                "temporal_mass_gate_threshold",
                                0.0,
                            ),
                            temporal_mass_gate_gqa_union=method_config.get(
                                "temporal_mass_gate_gqa_union",
                                False,
                            ),
                            use_preallocated_cache=should_use_preallocated_cache(
                                executed_path,
                                int(bundle.input_ids.shape[-1]),
                            ),
                            query_calibration_tokens=int(
                                args.official_query_tail_tokens
                            ),
                            fixed_packed_qmse_templates=(
                                fixed_packed_qmse_templates
                                if executed_path
                                in QKSIEVE_GLOBAL_WMMA_SAMPLED_METHODS
                                else None
                            ),
                            fixed_binarypc_projections=(
                                fixed_binarypc_projections
                                if executed_path
                                == BINARYPC_OFFLINE64_FULLTOPK_METHOD
                                else None
                            ),
                            prefill_query_record_sink=(
                                prefill_query_records
                                if trace_enabled
                                else None
                            ),
                            prefill_query_record_tokens=(
                                int(
                                    qk_trace_config[
                                        "prefill_query_tail_tokens"
                                    ]
                                )
                                if trace_enabled
                                else None
                            ),
                            sampled_quantile_sample_count=method_config.get(
                                "sampled_quantile_sample_count"
                            ),
                        )
                    if trace_enabled:
                        trace_file = qk_trace_path(
                            qk_trace_config["output_dir"],
                            example.task,
                            example.sample_id,
                            method,
                        )
                        torch.save(
                            {
                                "schema": "qksieve_generation_drift_trace_v1",
                                "model_name_or_path": args.model_name_or_path,
                                "task": example.task,
                                "sample_id": example.sample_id,
                                "method": method,
                                "score_mode": method_config["score_mode"],
                                "prompt_wrapper": args.prompt_wrapper,
                                "prompt_truncation_mode": (
                                    args.prompt_truncation_mode
                                ),
                                "prompt_tokens": int(
                                    bundle.input_ids.shape[-1]
                                ),
                                "prefix_tokens": int(bundle.query_start),
                                "suffix_tokens": int(
                                    bundle.suffix_token_count
                                ),
                                "query_calibration_tokens": int(
                                    args.official_query_tail_tokens
                                ),
                                "recorded_prefill_query_tail_tokens": int(
                                    qk_trace_config[
                                        "prefill_query_tail_tokens"
                                    ]
                                ),
                                "qk_metric_query_shrinkage": float(
                                    args.qk_metric_query_shrinkage
                                ),
                                "trace_layers": qk_trace_config["layers"],
                                "trace_steps": qk_trace_config["steps"],
                                "generated_ids": result["generated_ids"],
                                "prefill_query_tail": prefill_query_records,
                                "records": trace_records,
                            },
                            trace_file,
                        )
            score = lb.score_prediction(
                example.metric,
                result["prediction"],
                example.answers,
                example.all_classes,
                task=example.task,
            )
            row = {
                "task": example.task,
                "sample_id": example.sample_id,
                "method": method,
                "executed_path": executed_path,
                "gate_expected_generated_tokens": (
                    gate_decision["expected_generated_tokens"]
                    if gate_decision is not None
                    else ""
                ),
                "gate_predicted_decode_speedup": (
                    gate_decision["predicted_decode_speedup"]
                    if gate_decision is not None
                    else ""
                ),
                "metric": example.metric,
                "score": score,
                "prediction": result["prediction"],
                "answers": json.dumps(example.answers, ensure_ascii=False),
                "prompt_tokens": int(bundle.input_ids.shape[-1]),
                "prefix_tokens": bundle.query_start,
                "suffix_tokens": bundle.suffix_token_count,
                "generated_tokens": len(result["generated_ids"]),
                "configured_attention_fraction": method_config["budget_fractions"][-1],
                "configured_attention_tokens": method_config["attention_tokens"],
                "configured_candidate_fraction": method_config["candidate_fraction"],
                "configured_projection_dim": method_config["projection_dim"],
                "configured_score_mode": method_config["score_mode"],
                "configured_sampled_quantile_sample_count": (
                    method_config.get("sampled_quantile_sample_count", 0)
                ),
                "configured_index_bits_per_token": (
                    configured_index_bits_per_token(
                        method_config["score_mode"]
                    )
                ),
                "attention_link_ratio": result["attention_link_ratio"],
                "selected_history_fraction_mean": result.get(
                    "selected_history_fraction_mean", 0.0
                ),
                "selected_history_fraction_p95": result.get(
                    "selected_history_fraction_p95", 0.0
                ),
                "selected_history_fraction_max": result.get(
                    "selected_history_fraction_max", 0.0
                ),
                "selected_history_count_mean": result.get(
                    "selected_history_count_mean", 0.0
                ),
                "selected_history_count_p95": result.get(
                    "selected_history_count_p95", 0.0
                ),
                "selected_history_count_max": result.get(
                    "selected_history_count_max", 0.0
                ),
                "sampled_candidate_overflow_fraction": result.get(
                    "sampled_candidate_overflow_fraction", 0.0
                ),
                "sampled_quantile_fallback": result.get(
                    "sampled_quantile_fallback", 0.0
                ),
                "packed_qmse_index_bits_per_token": result.get(
                    "packed_qmse_index_bits_per_token", 0.0
                ),
                "packed_qmse_sample_count": result.get(
                    "packed_qmse_sample_count", 0.0
                ),
                "packed_qmse_value_sketch_rank": result.get(
                    "packed_qmse_value_sketch_rank", 0.0
                ),
                "packed_qmse_value_sketch_bits": result.get(
                    "packed_qmse_value_sketch_bits", 0.0
                ),
                "packed_qmse_value_sketch_executed": result.get(
                    "packed_qmse_value_sketch_executed", 0.0
                ),
                "packed_qmse_value_sketch_tail_alpha": result.get(
                    "packed_qmse_value_sketch_tail_alpha", 0.0
                ),
                "packed_qmse_debug_value_sketch_disabled": result.get(
                    "packed_qmse_debug_value_sketch_disabled", 0.0
                ),
                "packed_qmse_fused_query_prepare_requested": result.get(
                    "packed_qmse_fused_query_prepare_requested", 0.0
                ),
                "packed_qmse_fused_query_prepare_executed": result.get(
                    "packed_qmse_fused_query_prepare_executed", 0.0
                ),
                "packed_qmse_allocation_frozen_before_query": result.get(
                    "packed_qmse_allocation_frozen_before_query", 0.0
                ),
                "packed_qmse_fixed_template_active": result.get(
                    "packed_qmse_fixed_template_active", 0.0
                ),
                "public_selector_index_bits_per_token": result.get(
                    "public_selector_index_bits_per_token", 0.0
                ),
                "public_selector_dimension_count": result.get(
                    "public_selector_dimension_count", 0.0
                ),
                "public_selector_selected_page_count": result.get(
                    "public_selector_selected_page_count", 0.0
                ),
                "public_selector_is_page_granular": result.get(
                    "public_selector_is_page_granular", 0.0
                ),
                "public_selector_mean_value_correction": result.get(
                    "public_selector_mean_value_correction", 0.0
                ),
                "public_selector_local_window": result.get(
                    "public_selector_local_window", 0.0
                ),
                "public_selector_approximate_selected_mass": result.get(
                    "public_selector_approximate_selected_mass", 0.0
                ),
                "packed_index_ratio_of_full_kv": (
                    result.get("packed_qmse_index_bits_per_token", 0.0)
                    / 4096.0
                ),
                "exact_qk_ratio": result["exact_qk_ratio"],
                "estimated_retained_mass": result["estimated_retained_mass"],
                "temporal_reuse_rate": result["temporal_reuse_rate"],
                "gpu_kv_storage_ratio": result["gpu_kv_storage_ratio"],
                "scan_dimension_fraction": result["scan_dimension_fraction"],
                "diagnostics_enabled": args.collect_attention_stats,
                "prefill_seconds": result["prefill_seconds"],
                "query_seconds": result["query_seconds"],
                "decode_seconds": result["decode_seconds"],
                "index_build_seconds": result.get("index_build_seconds", 0.0),
                "qk_prebuild_seconds": result.get("qk_prebuild_seconds", 0.0),
                "qk_prebuild_layers": result.get("qk_prebuild_layers", 0),
                "qk_batched_allocation_layers": result.get(
                    "qk_batched_allocation_layers", 0
                ),
                "online_seconds": result["query_seconds"] + result["decode_seconds"],
                "total_seconds": result["prefill_seconds"]
                + result["query_seconds"]
                + result["decode_seconds"],
            }
            for name in TEMPORAL_TRACE_METRICS:
                row[name] = result.get(name, 0.0)
            rows.append(row)
            append_csv_row(results_path, row)
            completed.add(key)
            print(
                f"  {method}: score={score:.4f} links={float(result['attention_link_ratio']):.4f} "
                f"exact_qk={float(result['exact_qk_ratio']):.4f} "
                f"online={row['online_seconds']:.3f}s pred={result['prediction'][:80]!r}",
                flush=True,
            )
            empty_cuda_caches()

    summary = summarize(rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
