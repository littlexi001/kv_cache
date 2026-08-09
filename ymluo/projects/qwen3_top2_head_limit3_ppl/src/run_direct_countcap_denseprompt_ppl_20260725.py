from __future__ import annotations

import argparse
import cProfile
import csv
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_20newsgroups

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_controlled_public_kv_benchmark_v1 as lb  # noqa: E402
import run_head_top2_targeted_ppl_20260714 as sparse_attention  # noqa: E402
from preallocated_dynamic_cache_20260724 import (  # noqa: E402
    PreallocatedDynamicCache,
)
from run_critical_position_budget_probe_20260715 import (  # noqa: E402
    run_one_token,
)
from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    direct_countcap_target_count,
    export_active_packed_qmse_templates,
    head_qabs_sampled_mass_mode,
    head_top_fraction_mode,
    head_top_k_mode,
    install_active_binarypc_projections,
    install_active_packed_qmse_templates,
    install_llama_head_top_fraction_patch,
    load_model,
    parse_int_list,
    prefill_query_tail_mode,
    seed_packed_qmse_prefill_queries,
    set_attention_implementation,
)
from run_multitopic_lpcm_ppl_20260714 import (  # noqa: E402
    TOPICS,
    encode_topic_stream,
    make_bundle,
)


METHODS = (
    "full_attention",
    "exact_top2",
    "exact_top_fraction",
    "direct_countcap",
)
SUPPORTED_METHODS = METHODS + ("exact_top_k",)
DIRECT_SCORE_MODE = (
    "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_"
    "qprojscan_qkvsplitauto"
)
STRICT_PROXY_TOPK_SCORE_MODE = "pca_int4_chunked_logscale16_autosplit"
PACKED_QMSE_SCORE_MODE = "pca_hierarchical_autoqmsetotal15z_packed_direct"
PACKED_QMSE_OAS_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_packed_direct_oas"
)
PACKED_QMSE_QKMETRIC_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_direct"
)
PACKED_QMSE_QKMETRIC_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FREQUENCY_TIERED_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_freqtier12hot4_"
    "packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FREQUENCY_BLOCK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_freqblock16hot10_"
    "packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FREQUENCY_BLOCK32_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_freqblock32hot10_"
    "packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_COMPUTEAWARE_BLOCK32_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "cold84freqblock32hot15_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_COMPUTEAWARE_SAMPLED_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "cold84freqblock32hot15_sampled_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FREQSKIP50_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "freqskip50shard4recent256carry_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FREQSKIP60_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "freqskip60shard4recent256carry_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_LOW192_SCORE_MODE = (
    "pca_hierarchical_fixed441_qkmetric_packed_direct"
)
PACKED_QMSE_QKMETRIC_LOW192_UNBIASED_SCORE_MODE = (
    "pca_hierarchical_fixed441_qkmetric_unbiased_packed_direct"
)
PACKED_QMSE_QKMETRIC_FIXED441_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed441_qkmetric_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED4421_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed4421_qkmetric_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED4221_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed4221_qkmetric_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED440_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed440_qkmetric_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED440_FULL_PREROPE_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed440_qkmetric_fullprerope_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED440_FULL_PREROPE_LOCALSINK_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed440_qkmetric_fullprerope_"
    "localsink_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED441_FULL_PREROPE_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed441_qkmetric_fullprerope_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED420_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed420_qkmetric_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED410_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED410_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH16_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch16i4shared_unbiased_packed_direct"
)
PACKED_QMSE_QKMETRIC_FP16X2_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixedfp16x2_qkmetric_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED211_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed211_qkmetric_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED220_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed220_qkmetric_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED420_POST2X_PRERERANK_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed420_qkmetric_post2xprererank_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED410_POST2X_PRERERANK_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_post2xprererank_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO08_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_"
    "post2xprererank_l00to08_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED410_POST2X_DUALMASS_L00TO08_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_"
    "post2xdualmass_l00to08_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO02_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_"
    "post2xprererank_l00to02_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO05_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_"
    "post2xprererank_l00to05_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L09TO17_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_"
    "post2xprererank_l09to17_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO17_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_"
    "post2xprererank_l00to17_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L18TO26_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_"
    "post2xprererank_l18to26_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO26_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_"
    "post2xprererank_l00to26_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L27TO35_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed410_qkmetric_"
    "post2xprererank_l27to35_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED420_POST2X_BOUNDARY75_PRERERANK_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed420_qkmetric_"
    "post2xboundary75prererank_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED420_FULL_PREROPE_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed420_qkmetric_fullprerope_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED420_FULL_PREROPE_LOCALSINK_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed420_qkmetric_fullprerope_"
    "localsink_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED420_PREROPE32INT2_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed420_qkmetric_prerope32int2_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED420_PREROPE32INT4_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed420_qkmetric_prerope32int4_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED420_PREROPE32ADAPTIVE_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed420_qkmetric_prerope32adaptive_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED400_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed400_qkmetric_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED400_MINIFLOAT_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed400_qkmetric_minifloat_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED200_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed200_qkmetric_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_FIXED420_MINIFLOAT_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed420_qkmetric_minifloat_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_QFUSED_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_packed_fulltopk"
)
PACKED_QMSE_QKMETRIC_QFUSED_UNBIASED_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_unbiased_"
    "packed_direct"
)
PACKED_QMSE_QKMETRIC_QFUSED_GQA4_UNBIASED_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_unbiased_"
    "packed_direct"
)
PACKED_QMSE_QKMETRIC_QFUSED_GQA4_TIGHTCAP_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_unbiased_"
    "tightcap_packed_direct"
)
PACKED_QMSE_QKMETRIC_QFUSED_GQA4_KAPPEND_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_kappend_"
    "unbiased_packed_direct"
)
PACKED_QMSE_QKMETRIC_QFUSED_GQA4_KAPPEND_TIGHTCAP_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_kappend_"
    "unbiased_tightcap_packed_direct"
)
PACKED_QMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_unbiased_packed_direct"
)
PACKED_QMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_TIGHTCAP_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_unbiased_tightcap_packed_direct"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_unbiased_packed_direct"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SAMPLEMASS_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_samplemass_unbiased_packed_direct"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_PROXYMASS_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_proxymass_unbiased_packed_direct"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_MEANTAIL_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_meantail_unbiased_packed_direct"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_TILTTAIL16_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_tilttail16_unbiased_packed_direct"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_CONDTAIL32_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_condtail32b128i8_unbiased_packed_direct"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_CONDTAIL16_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_condtail16b512i8_unbiased_packed_direct"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_CONDTAIL8_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_condtail8b1024i8_unbiased_packed_direct"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH8_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch8i4shared_unbiased_packed_direct"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH16_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch16i4shared_unbiased_packed_direct"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_MASSLADDER90_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch16i4shared_massladder900_unbiased_packed_direct"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_MASSLADDER95_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch16i4shared_massladder950_unbiased_packed_direct"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH32_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch32i4shared_unbiased_packed_direct"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_LAYER0R128_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_layer0r128_packed_fulltopk"
)
PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH8TO32_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch8to32i4shared_tau40m_unbiased_packed_direct"
)
PACKED_QMSE_QKMETRIC_FULLTOPK_FP16_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk_fp16"
)
PACKED_QMSE_QKMETRIC_QSCALE_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qscale_packed_direct"
)
PACKED_QMSE_QKMETRIC_QSCALE_MEANBIAS_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qscale_"
    "meanbias_packed_direct"
)
PACKED_QMSE_QKMETRIC_CENTERED_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_centered_packed_direct"
)
PACKED_SHAREDTAIL_QKMETRIC_CENTERED_SCORE_MODE = (
    "pca_hierarchical_sharedtail240_qkmetric_centered_packed_direct"
)
FIER_RTN1_G32_FULLTOPK_SCORE_MODE = "fier_rtn1_g32_fulltopk"
FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE = (
    "fier_rtn1_g32_packed_fulltopk"
)
FIER_RTN1_G32_SAMPLED_PACKED_SCORE_MODE = (
    "fier_rtn1_g32_sampled_packed_direct"
)
PACKED_KEYPCA_UNIFORM1_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed11111111_packed_fulltopk"
)
PACKED_QKBALANCED_UNIFORM1_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed11111111_qkmetric_packed_fulltopk"
)
PACKED_RANDOM_UNIFORM1_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed11111111_random_packed_fulltopk"
)
PACKED_KEYPCA_AUTOKEY_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_BLOCKMASS32_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "blockmass32_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_BLOCKMASS64_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "blockmass64_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_BLOCKMASS128_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "blockmass128_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_BLOCKMASS256_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "blockmass256_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_BLOCKMASS512_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "blockmass512_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_BLOCKCALMASS64_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "blockcalmass64_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_BLOCKSHAREDCALMASS64_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "blocksharedcalmass64_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_BLOCKSHAREDCALMASS128_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "blocksharedcalmass128_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH8I4_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch8i4shared_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR900_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_massfloor900_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR950_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_massfloor950_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR975_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_massfloor975_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK8_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk8_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_RSSREL5M_S1_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "rssrel5m_safety1_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_RSSREL5M_S2_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "rssrel5m_safety2_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALFLOORRSS25E4_S1_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "globalfloorrss25e4_safety1_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_PREFIXRSS25E4_S1_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "prefixrss25e4_safety1_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALALLOC_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "globalalloc_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALCAL256_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "globalcal256_globalalloc_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALANCHOR30_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "globalanchor30_globalalloc_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALANCHOR50_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "globalanchor50_globalalloc_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALANCHOR70_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "globalanchor70_globalalloc_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH32I4_WOMETRIC_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch32i4shared_wometric_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_CONDRES8_GLOBAL_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_condres8global_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR950_AFFINERES_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_massfloor950_"
    "affineresglobal_packed_fulltopk"
)
PACKED_QKSIEVE_QKMSE_DUALMASS975_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "dualmass975_globalcal256_affineresglobal_packed_fulltopk"
)
PACKED_QKSIEVE_QMSE_DUALMASS975_NOAFFINE_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "dualmass975_globalcal256_packed_fulltopk"
)
PACKED_QKSIEVE_QMSE_OAS_DUALMASS975_NOAFFINE_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "dualmass975_globalcal256_packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_DUALMASS975_DIAGONAL_NOAFFINE_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_riskdiag_"
    "dualmass975_globalcal256_packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_PREFIXRSS25E4_S1_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "prefixrss25e4_safety1_packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_PREFIXRSS25E4_S2_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "prefixrss25e4_safety2_packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_RATE23_PREFIXRSS25E4_S2_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal23z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "prefixrss25e4_safety2_packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_CROSSING_BERNSTEIN_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_keyrisk4_"
    "crossbernstein99_cal256_packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_EMPIRICAL_CROSSING_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_keyrisk4_"
    "crossempirical99_cal256_packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_EMPIRICAL_CROSSING_KEEPUNION_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_keyrisk4_"
    "crossempirical99_cal256_keepunion_packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_CONDRES8_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_condres8global_"
    "packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_CONDRES8_WIENER_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_condres8wienerglobal_"
    "packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_CONDRES8_QUERY_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_condres8queryglobal_"
    "packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_CONDRES8_SAFEQUERY_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_condres8safequeryglobal_"
    "packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_BLOCKCONDRES8_R8_M8_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch8i4shared_wometric_blockcondres8b256m8global_"
    "packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SAMPLED_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch16i4shared_wometric_unbiased_packed_direct_oas"
)
PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SORTED_SAMPLED_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_valuesketch16i4shared_wometric_sortcompact_unbiased_"
    "packed_direct_oas"
)
PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH8_WOMETRIC_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch8i4shared_wometric_packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH12_WOMETRIC_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch12i4shared_wometric_packed_fulltopk_oas"
)
PACKED_QKSIEVE_QMSE_OAS_CONDRES8_EMPIRICAL_CROSSING_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autoqmsetotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_condres8global_keyrisk4_"
    "crossempirical99_cal256_packed_fulltopk_oas"
)
PACKED_QKSIEVE_KEYMSE_DUALMASS975_NOAFFINE_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "dualmass975_globalcal256_packed_fulltopk"
)
PACKED_QKSIEVE_KEYMSE_DUALMASS975_AFFINE_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_residualrisk4_"
    "dualmass975_globalcal256_affineresglobal_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR950_CONDRES8_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch16i4shared_wometric_massfloor950_"
    "condres8global_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_VALUESKETCH32I4_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_"
    "valuesketch32i4shared_packed_fulltopk"
)
PACKED_QKBALANCED_AUTOKEY_REALLOC_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_realloc_packed_fulltopk"
)
PACKED_QKBALANCED_FIXED41111100_FULLTOPK_SCORE_MODE = (
    "pca_hierarchical_fixed41111100_qkmetric_packed_fulltopk"
)
PACKED_QMSE_SCORE_MODES = {
    PACKED_QMSE_SCORE_MODE,
    PACKED_QMSE_OAS_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FREQUENCY_TIERED_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FREQUENCY_BLOCK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FREQUENCY_BLOCK32_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_COMPUTEAWARE_BLOCK32_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_COMPUTEAWARE_SAMPLED_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FREQSKIP50_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FREQSKIP60_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_LOW192_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_LOW192_UNBIASED_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED441_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED4421_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED4221_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED440_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED440_FULL_PREROPE_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED440_FULL_PREROPE_LOCALSINK_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED441_FULL_PREROPE_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED420_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH16_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FP16X2_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED211_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED220_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED420_POST2X_PRERERANK_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_POST2X_PRERERANK_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO02_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO05_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO08_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_POST2X_DUALMASS_L00TO08_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L09TO17_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO17_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L18TO26_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO26_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L27TO35_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED420_POST2X_BOUNDARY75_PRERERANK_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED420_FULL_PREROPE_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED420_FULL_PREROPE_LOCALSINK_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED420_PREROPE32INT2_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED420_PREROPE32INT4_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED420_PREROPE32ADAPTIVE_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED400_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED400_MINIFLOAT_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED200_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED420_MINIFLOAT_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_QFUSED_UNBIASED_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_QFUSED_GQA4_UNBIASED_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_QFUSED_GQA4_TIGHTCAP_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_QFUSED_GQA4_KAPPEND_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_QFUSED_GQA4_KAPPEND_TIGHTCAP_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_TIGHTCAP_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SAMPLEMASS_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_PROXYMASS_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_MEANTAIL_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_TILTTAIL16_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_CONDTAIL32_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_CONDTAIL16_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_CONDTAIL8_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH8_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH16_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_MASSLADDER90_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_MASSLADDER95_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH32_SCORE_MODE,
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH8TO32_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_QFUSED_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FULLTOPK_FP16_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_QSCALE_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_QSCALE_MEANBIAS_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_CENTERED_SCORE_MODE,
    PACKED_SHAREDTAIL_QKMETRIC_CENTERED_SCORE_MODE,
    PACKED_KEYPCA_UNIFORM1_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_UNIFORM1_FULLTOPK_SCORE_MODE,
    PACKED_RANDOM_UNIFORM1_FULLTOPK_SCORE_MODE,
    PACKED_KEYPCA_AUTOKEY_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_BLOCKMASS32_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_BLOCKMASS64_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_BLOCKMASS128_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_BLOCKMASS256_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_BLOCKMASS512_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_BLOCKCALMASS64_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_BLOCKSHAREDCALMASS64_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_BLOCKSHAREDCALMASS128_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH8I4_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR900_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR950_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR975_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK8_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_RSSREL5M_S1_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_RSSREL5M_S2_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALFLOORRSS25E4_S1_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_PREFIXRSS25E4_S1_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALALLOC_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALCAL256_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALANCHOR30_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALANCHOR50_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALANCHOR70_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH32I4_WOMETRIC_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_CONDRES8_GLOBAL_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR950_AFFINERES_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QKMSE_DUALMASS975_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_DUALMASS975_NOAFFINE_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_DUALMASS975_NOAFFINE_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_DUALMASS975_DIAGONAL_NOAFFINE_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_PREFIXRSS25E4_S1_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_PREFIXRSS25E4_S2_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_RATE23_PREFIXRSS25E4_S2_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_CROSSING_BERNSTEIN_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_EMPIRICAL_CROSSING_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_EMPIRICAL_CROSSING_KEEPUNION_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_CONDRES8_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_CONDRES8_WIENER_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_CONDRES8_QUERY_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_CONDRES8_SAFEQUERY_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_BLOCKCONDRES8_R8_M8_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SAMPLED_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SORTED_SAMPLED_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH12_WOMETRIC_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH8_WOMETRIC_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_CONDRES8_EMPIRICAL_CROSSING_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_KEYMSE_DUALMASS975_NOAFFINE_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_KEYMSE_DUALMASS975_AFFINE_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR950_CONDRES8_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH32I4_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_LAYER0R128_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_REALLOC_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_FIXED41111100_FULLTOPK_SCORE_MODE,
    FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE,
    FIER_RTN1_G32_SAMPLED_PACKED_SCORE_MODE,
}
PACKED_PREFILL_QUERY_SCORE_MODES = PACKED_QMSE_SCORE_MODES - {
    PACKED_KEYPCA_UNIFORM1_FULLTOPK_SCORE_MODE,
    PACKED_RANDOM_UNIFORM1_FULLTOPK_SCORE_MODE,
    PACKED_KEYPCA_AUTOKEY_FULLTOPK_SCORE_MODE,
    FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE,
    FIER_RTN1_G32_SAMPLED_PACKED_SCORE_MODE,
}
PACKED_FREQUENCY_PREFILL_QUERY_SCORE_MODES = {
    PACKED_QMSE_QKMETRIC_FREQUENCY_TIERED_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FREQUENCY_BLOCK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FREQUENCY_BLOCK32_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_COMPUTEAWARE_BLOCK32_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_COMPUTEAWARE_SAMPLED_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FREQSKIP50_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FREQSKIP60_SCORE_MODE,
}
MIXED_TOPIC_POOLS = {
    "sports_both": (
        "rec.sport.baseball",
        "rec.sport.hockey",
    ),
    "mixed_a": (
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
        "sci.crypt",
    ),
    "mixed_b": (
        "sci.electronics",
        "sci.med",
        "sci.space",
        "misc.forsale",
        "talk.politics.guns",
        "talk.politics.mideast",
        "talk.politics.misc",
        "talk.religion.misc",
        "alt.atheism",
        "soc.religion.christian",
    ),
}


def tail_resolution_sample_count(
    target_tail_count: int,
    target_fraction: float,
    *,
    alignment: int = 256,
    minimum: int = 256,
    maximum: int = 8192,
) -> int:
    """Choose an aligned sample count with enough expected tail anchors."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dense-prompt PPL for direct length-capped Key-PCA attention."
        )
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--topics", default="sports,medicine")
    parser.add_argument("--window_indices", default="0,1,2")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--window_stride_tokens", type=int, default=32_512)
    parser.add_argument(
        "--target_anchor_tokens",
        type=int,
        default=0,
        help=(
            "When positive, all lengths predict targets beginning at this "
            "absolute stream position plus window * stride."
        ),
    )
    parser.add_argument("--direct_fraction", type=float, default=0.06)
    parser.add_argument(
        "--exact_fraction",
        type=float,
        default=0.02,
        help=(
            "Per-head fraction for the exact_top_fraction diagnostic. "
            "exact_top2 remains fixed at 0.02 for compatibility."
        ),
    )
    parser.add_argument(
        "--exact_tokens",
        type=int,
        default=1280,
        help=(
            "Absolute historical-token budget for the exact_top_k "
            "full-FP16-QK oracle diagnostic."
        ),
    )
    parser.add_argument("--direct_min_tokens", type=int, default=256)
    parser.add_argument("--direct_max_tokens", type=int, default=1280)
    parser.add_argument("--projection_dim", type=int, default=48)
    parser.add_argument("--sample_count", type=int, default=256)
    parser.add_argument("--candidate_overfetch", type=float, default=1.0)
    parser.add_argument("--protect_recent_tokens", type=int, default=0)
    parser.add_argument(
        "--direct_score_mode",
        choices=(
            DIRECT_SCORE_MODE,
            STRICT_PROXY_TOPK_SCORE_MODE,
            PACKED_QMSE_SCORE_MODE,
            PACKED_QMSE_OAS_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_FULLTOPK_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_FREQUENCY_TIERED_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_FREQUENCY_BLOCK_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_FREQUENCY_BLOCK32_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_COMPUTEAWARE_BLOCK32_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_COMPUTEAWARE_SAMPLED_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_FREQSKIP50_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_FREQSKIP60_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_LOW192_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_LOW192_UNBIASED_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_QFUSED_UNBIASED_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_QFUSED_GQA4_UNBIASED_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_QFUSED_GQA4_TIGHTCAP_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_QFUSED_GQA4_KAPPEND_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_QFUSED_GQA4_KAPPEND_TIGHTCAP_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_TIGHTCAP_SCORE_MODE,
            PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE,
            PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SAMPLEMASS_SCORE_MODE,
            PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_PROXYMASS_SCORE_MODE,
            PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH8_SCORE_MODE,
            PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH16_SCORE_MODE,
            PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH32_SCORE_MODE,
            PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH8TO32_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_QFUSED_FULLTOPK_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_FULLTOPK_FP16_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_FP16X2_FULLTOPK_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_QSCALE_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_QSCALE_MEANBIAS_SCORE_MODE,
            PACKED_QMSE_QKMETRIC_CENTERED_SCORE_MODE,
            PACKED_SHAREDTAIL_QKMETRIC_CENTERED_SCORE_MODE,
            FIER_RTN1_G32_FULLTOPK_SCORE_MODE,
            FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE,
            FIER_RTN1_G32_SAMPLED_PACKED_SCORE_MODE,
            PACKED_KEYPCA_UNIFORM1_FULLTOPK_SCORE_MODE,
            PACKED_QKBALANCED_UNIFORM1_FULLTOPK_SCORE_MODE,
            PACKED_RANDOM_UNIFORM1_FULLTOPK_SCORE_MODE,
            PACKED_KEYPCA_AUTOKEY_FULLTOPK_SCORE_MODE,
            PACKED_QKBALANCED_AUTOKEY_FULLTOPK_SCORE_MODE,
            PACKED_QKSIEVE_QMSE_OAS_PREFIXRSS25E4_S1_FULLTOPK_SCORE_MODE,
            PACKED_QKSIEVE_QMSE_OAS_PREFIXRSS25E4_S2_FULLTOPK_SCORE_MODE,
            PACKED_QKSIEVE_QMSE_OAS_RATE23_PREFIXRSS25E4_S2_FULLTOPK_SCORE_MODE,
            PACKED_QKSIEVE_QMSE_OAS_CROSSING_BERNSTEIN_FULLTOPK_SCORE_MODE,
            PACKED_QKSIEVE_QMSE_OAS_EMPIRICAL_CROSSING_FULLTOPK_SCORE_MODE,
            PACKED_QKSIEVE_QMSE_OAS_EMPIRICAL_CROSSING_KEEPUNION_FULLTOPK_SCORE_MODE,
            PACKED_QKSIEVE_QMSE_OAS_CONDRES8_FULLTOPK_SCORE_MODE,
            PACKED_QKSIEVE_QMSE_OAS_CONDRES8_WIENER_FULLTOPK_SCORE_MODE,
            PACKED_QKSIEVE_QMSE_OAS_CONDRES8_QUERY_FULLTOPK_SCORE_MODE,
            PACKED_QKSIEVE_QMSE_OAS_CONDRES8_SAFEQUERY_FULLTOPK_SCORE_MODE,
            PACKED_QKSIEVE_QMSE_OAS_BLOCKCONDRES8_R8_M8_FULLTOPK_SCORE_MODE,
            PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_FULLTOPK_SCORE_MODE,
            PACKED_QKSIEVE_QMSE_OAS_CONDRES8_EMPIRICAL_CROSSING_FULLTOPK_SCORE_MODE,
        ),
        default=DIRECT_SCORE_MODE,
        help=(
            "Candidate selector. The default estimates a threshold from sampled "
            "proxy scores; strict proxy top-k materializes all low-bit proxy "
            "scores and selects exactly the configured per-head count."
        ),
    )
    parser.add_argument(
        "--qk_metric_query_shrinkage",
        type=float,
        default=0.5,
        help=(
            "Isotropic shrinkage applied to the empirical query second moment "
            "when constructing the QK-balanced biorthogonal coordinates."
        ),
    )
    parser.add_argument(
        "--packed_qmse_template_in",
        type=Path,
        default=None,
        help=(
            "Load frozen per-layer QKSieve transforms and bit allocations. "
            "The template is installed before the first sparse decode step."
        ),
    )
    parser.add_argument(
        "--packed_qmse_template_out",
        type=Path,
        default=None,
        help=(
            "Export the online per-layer QKSieve transforms and bit "
            "allocations after evaluation."
        ),
    )
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument(
        "--cache_mode",
        choices=("dynamic", "preallocated", "auto"),
        default="dynamic",
        help=(
            "KV cache allocation policy. auto switches to preallocated cache "
            "at --preallocated_cache_min_tokens."
        ),
    )
    parser.add_argument(
        "--preallocated_cache_min_tokens",
        type=int,
        default=14_000,
    )
    parser.add_argument(
        "--dataset_cache_dir",
        default="/home/fdong/ymluo/datasets/sklearn",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--collect_logit_stability",
        action="store_true",
        help=(
            "Retain Full logits on CPU for each case and report exact "
            "Full-versus-sparse KL, top-1 agreement, and margin stability."
        ),
    )
    return parser.parse_args()


def parse_methods(spec: str) -> list[str]:
    methods = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if not methods or unknown:
        raise ValueError(f"unsupported methods: {unknown}")
    return list(dict.fromkeys(methods))


def parse_topics(spec: str) -> list[str]:
    topics = [item.strip() for item in spec.split(",") if item.strip()]
    supported = set(TOPICS) | set(MIXED_TOPIC_POOLS)
    unknown = sorted(set(topics) - supported)
    if not topics or unknown:
        raise ValueError(f"unsupported topics: {unknown}")
    return list(dict.fromkeys(topics))


def encode_mixed_topic_stream(
    tokenizer: Any,
    topic: str,
    required_tokens: int,
    cache_dir: str,
    seed: int,
    repeat_documents: bool = False,
) -> list[int]:
    categories = MIXED_TOPIC_POOLS[topic]
    dataset = fetch_20newsgroups(
        subset="train",
        categories=list(categories),
        remove=("headers", "footers", "quotes"),
        data_home=cache_dir,
        shuffle=False,
    )
    documents = [
        text.strip()
        for text in dataset.data
        if len(text.strip()) >= 200
    ]
    stream: list[int] = []
    separator = "\n\n---\n\n"
    cycle = 0
    while len(stream) < required_tokens:
        cycle_documents = list(documents)
        random.Random(seed + cycle).shuffle(cycle_documents)
        for document in cycle_documents:
            stream.extend(
                tokenizer(
                    separator + document,
                    add_special_tokens=False,
                )["input_ids"]
            )
            if len(stream) >= required_tokens:
                break
        if not repeat_documents:
            break
        if not cycle_documents:
            break
        cycle += 1
    if len(stream) < required_tokens:
        raise RuntimeError(
            f"{topic} has only {len(stream)} usable tokens, "
            f"need {required_tokens}"
        )
    return stream[:required_tokens]


def encode_evaluation_stream(
    tokenizer: Any,
    topic: str,
    required_tokens: int,
    cache_dir: str,
    seed: int,
    repeat_documents: bool = False,
) -> list[int]:
    if topic in MIXED_TOPIC_POOLS:
        return encode_mixed_topic_stream(
            tokenizer,
            topic,
            required_tokens,
            cache_dir,
            seed,
            repeat_documents=repeat_documents,
        )
    return encode_topic_stream(
        tokenizer,
        TOPICS[topic],
        required_tokens,
        cache_dir,
        seed,
        repeat_documents=repeat_documents,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def token_nll(logits: torch.Tensor, label: int) -> float:
    return -float(F.log_softmax(logits[0].float(), dim=-1)[label].item())


def logit_stability_metrics(
    full_logits: torch.Tensor,
    sparse_logits: torch.Tensor,
    label: int,
) -> dict[str, Any]:
    sparse = sparse_logits.reshape(-1).float()
    full = full_logits.reshape(-1).to(sparse.device, torch.float32)
    if full.shape != sparse.shape:
        raise ValueError("full and sparse logits must have the same shape")
    if not 0 <= label < full.numel():
        raise ValueError("label is outside the vocabulary")

    full_top = torch.topk(full, k=2)
    sparse_top1 = int(torch.argmax(sparse).item())
    full_top1 = int(full_top.indices[0].item())
    full_margin = float((full_top.values[0] - full_top.values[1]).item())

    full_logp = F.log_softmax(full, dim=-1)
    sparse_logp = F.log_softmax(sparse, dim=-1)
    full_prob = full_logp.exp()
    sparse_prob = sparse_logp.exp()
    kl_full_sparse = float(
        torch.sum(full_prob * (full_logp - sparse_logp)).item()
    )
    midpoint = 0.5 * (full_prob + sparse_prob)
    log_midpoint = midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny).log()
    js_divergence = 0.5 * (
        torch.sum(full_prob * (full_logp - log_midpoint))
        + torch.sum(sparse_prob * (sparse_logp - log_midpoint))
    )

    delta = sparse - full
    shift_invariant_range = float((delta.max() - delta.min()).item())
    target_nll_delta = float(
        (full_logp[label] - sparse_logp[label]).item()
    )
    kl_range_upper_bound = shift_invariant_range**2 / 8.0
    return {
        "full_top1_id": full_top1,
        "sparse_top1_id": sparse_top1,
        "top1_agreement": int(full_top1 == sparse_top1),
        "full_top1_margin": full_margin,
        "shift_invariant_logit_delta_range": shift_invariant_range,
        "margin_certificate_satisfied": int(
            full_margin > shift_invariant_range
        ),
        "kl_full_to_sparse": kl_full_sparse,
        "kl_range_upper_bound": kl_range_upper_bound,
        "kl_range_bound_satisfied": int(
            kl_full_sparse <= kl_range_upper_bound + 1.0e-6
        ),
        "js_divergence": float(js_divergence.item()),
        "target_nll_delta": target_nll_delta,
        "target_nll_range_bound_satisfied": int(
            abs(target_nll_delta) <= shift_invariant_range + 1.0e-6
        ),
    }


def active_candidate_count_stats() -> tuple[float, int, int] | None:
    states = sparse_attention._ACTIVE_QABS_PCA_STATES
    if not states:
        return None
    counts = [
        state.get(
            "last_consumed_candidate_counts",
            state["last_sampled_candidate_counts"],
        ).reshape(-1)
        for state in states.values()
        if "last_sampled_candidate_counts" in state
    ]
    if not counts:
        return None
    by_device: dict[torch.device, list[torch.Tensor]] = {}
    for tensor in counts:
        by_device.setdefault(tensor.device, []).append(tensor)
    device_stats = []
    for tensors in by_device.values():
        packed = torch.cat(tensors).float()
        device_stats.append(
            (
                float(packed.sum().item()),
                int(packed.numel()),
                int(packed.min().item()),
                int(packed.max().item()),
            )
        )
    total = sum(item[0] for item in device_stats)
    count = sum(item[1] for item in device_stats)
    return (
        total / count,
        min(item[2] for item in device_stats),
        max(item[3] for item in device_stats),
    )


def active_candidate_count_stats_by_layer() -> list[dict[str, float | int]]:
    """Return optional per-layer candidate quantiles for mechanism probes."""

    rows: list[dict[str, float | int]] = []
    for layer, state in sorted(sparse_attention._ACTIVE_QABS_PCA_STATES.items()):
        counts = state.get(
            "last_consumed_candidate_counts",
            state.get("last_sampled_candidate_counts"),
        )
        if not isinstance(counts, torch.Tensor):
            continue
        packed = counts.reshape(-1).float()
        quantiles = torch.quantile(
            packed,
            torch.tensor(
                [0.5, 0.9, 0.99],
                dtype=packed.dtype,
                device=packed.device,
            ),
        )
        rows.append(
            {
                "layer": int(layer),
                "mean": float(packed.mean().item()),
                "min": int(packed.min().item()),
                "p50": float(quantiles[0].item()),
                "p90": float(quantiles[1].item()),
                "p99": float(quantiles[2].item()),
                "max": int(packed.max().item()),
                "counts": [int(item) for item in packed.cpu().tolist()],
            }
        )
    return rows


def active_candidate_overflow_stats() -> tuple[float, int] | None:
    states = sparse_attention._ACTIVE_QABS_PCA_STATES
    if not states:
        return None
    overflows = [
        state["last_sampled_candidate_overflow"].reshape(-1)
        for state in states.values()
        if "last_sampled_candidate_overflow" in state
    ]
    if not overflows:
        return None
    by_device: dict[torch.device, list[torch.Tensor]] = {}
    for tensor in overflows:
        by_device.setdefault(tensor.device, []).append(tensor)
    overflow_count = 0
    head_count = 0
    for tensors in by_device.values():
        packed = torch.cat(tensors)
        overflow_count += int(packed.sum().item())
        head_count += int(packed.numel())
    return overflow_count / max(1, head_count), overflow_count


def active_crossing_stats() -> tuple[float, int, float] | None:
    """Aggregate crossing diagnostics once after the full model forward."""

    states = sparse_attention._ACTIVE_QABS_PCA_STATES
    if not states:
        return None
    by_device: dict[
        torch.device, list[tuple[torch.Tensor, torch.Tensor]]
    ] = {}
    for state in states.values():
        counts = state.get("qksieve_crossing_rescue_counts")
        expected = state.get("qksieve_crossing_expected_crossings")
        if not isinstance(counts, torch.Tensor) or not isinstance(
            expected, torch.Tensor
        ):
            continue
        by_device.setdefault(counts.device, []).append(
            (counts.reshape(-1), expected.reshape(-1))
        )
    if not by_device:
        return None
    count_sum = 0.0
    expected_sum = 0.0
    element_count = 0
    maximum = 0
    for pairs in by_device.values():
        counts = torch.cat([pair[0] for pair in pairs]).float()
        expected = torch.cat([pair[1] for pair in pairs]).float()
        count_sum += float(counts.sum().item())
        expected_sum += float(expected.sum().item())
        element_count += int(counts.numel())
        maximum = max(maximum, int(counts.max().item()))
    return (
        count_sum / max(1, element_count),
        maximum,
        expected_sum / max(1, element_count),
    )


def active_value_refinement_stats() -> tuple[float, int, int] | None:
    states = sparse_attention._ACTIVE_QABS_PCA_STATES
    if not states:
        return None
    flags = [
        state["packed_qmse_value_refinement_flags"].reshape(-1)
        for state in states.values()
        if bool(state.get("packed_qmse_value_sketch_progressive", False))
        and "packed_qmse_value_refinement_flags" in state
    ]
    if not flags:
        return None
    by_device: dict[torch.device, list[torch.Tensor]] = {}
    for tensor in flags:
        by_device.setdefault(tensor.device, []).append(tensor)
    refinement_count = 0
    head_count = 0
    for tensors in by_device.values():
        packed = torch.cat(tensors)
        refinement_count += int(packed.sum().item())
        head_count += int(packed.numel())
    return (
        refinement_count / max(1, head_count),
        refinement_count,
        head_count,
    )


def window_bounds(
    history_tokens: int,
    window: int,
    window_stride_tokens: int,
    target_anchor_tokens: int = 0,
) -> tuple[int, int]:
    if history_tokens <= 0 or window < 0 or window_stride_tokens <= 0:
        raise ValueError("invalid history/window/stride")
    if target_anchor_tokens > 0:
        history_end = target_anchor_tokens + window * window_stride_tokens
        history_start = history_end - history_tokens
        if history_start < 0:
            raise ValueError("target anchor must be at least history_tokens")
        return history_start, history_end
    history_start = window * window_stride_tokens
    return history_start, history_start + history_tokens


def should_use_preallocated_cache(
    cache_mode: str,
    history_tokens: int,
    preallocated_cache_min_tokens: int,
) -> bool:
    if cache_mode not in {"dynamic", "preallocated", "auto"}:
        raise ValueError(f"unsupported cache_mode: {cache_mode}")
    if history_tokens <= 0:
        raise ValueError("history_tokens must be positive")
    if preallocated_cache_min_tokens <= 0:
        raise ValueError("preallocated_cache_min_tokens must be positive")
    return cache_mode == "preallocated" or (
        cache_mode == "auto"
        and history_tokens >= preallocated_cache_min_tokens
    )


def build_initial_cache(
    cache_mode: str,
    history_tokens: int,
    max_new_tokens: int,
    preallocated_cache_min_tokens: int,
) -> PreallocatedDynamicCache | None:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not should_use_preallocated_cache(
        cache_mode,
        history_tokens,
        preallocated_cache_min_tokens,
    ):
        return None
    return PreallocatedDynamicCache(
        max_cache_len=(
            history_tokens
            + max_new_tokens
            + 8
            + int(os.environ.get("QKSIEVE_GRAPH_SUFFIX_CAPACITY", "0"))
        ),
    )


@torch.inference_mode()
def dense_prompt(
    model: torch.nn.Module,
    tokenizer: Any,
    history: list[int],
    input_device: torch.device,
    prefill_chunk_tokens: int,
    cache_mode: str,
    preallocated_cache_min_tokens: int,
    max_new_tokens: int,
) -> tuple[Any, torch.Tensor, float]:
    if len(history) < 2:
        raise ValueError("history must contain at least two tokens")
    bundle, _ = make_bundle(tokenizer, history[:-1], 16)
    initial_cache = build_initial_cache(
        cache_mode,
        len(history),
        max_new_tokens,
        preallocated_cache_min_tokens,
    )
    set_attention_implementation(model, "sdpa")
    with head_top_fraction_mode(None), head_qabs_sampled_mass_mode(None):
        cache, prefill_seconds = lb.prefill_prefix(
            model,
            bundle,
            input_device,
            prefill_chunk_tokens,
            initial_cache=initial_cache,
        )
        cache, logits, last_seconds, _ = run_one_token(
            model,
            int(history[-1]),
            cache,
            len(history) - 1,
            input_device,
            collect_attention_stats=False,
        )
    return cache, logits, prefill_seconds + last_seconds


@torch.inference_mode()
def benchmark_fixed_position_model_cudagraph(
    model: torch.nn.Module,
    token_id: int,
    past_key_values: Any,
    position: int,
    input_device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Capture one fixed-position Decode step and verify replayed logits."""
    if input_device.type != "cuda":
        raise RuntimeError("the model CUDA Graph probe requires a CUDA device")
    if warmup < 1 or iterations < 1:
        raise ValueError("CUDA Graph warmup and iterations must be positive")
    crop = getattr(past_key_values, "crop", None)
    if not callable(crop):
        raise TypeError("the CUDA Graph probe requires a crop-capable cache")

    static_input_ids = torch.tensor(
        [[int(token_id)]], dtype=torch.long, device=input_device
    )
    static_cache_position = torch.tensor(
        [int(position)], dtype=torch.long, device=input_device
    )
    model_dtype = next(model.parameters()).dtype
    static_attention_mask = {
        "full_attention": torch.zeros(
            1,
            1,
            1,
            position + 1,
            dtype=model_dtype,
            device=input_device,
        )
    }

    def forward() -> Any:
        return lb.model_forward(
            model,
            {
                "input_ids": static_input_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
                "cache_position": static_cache_position,
                "attention_mask": static_attention_mask,
            },
        )

    for _ in range(2):
        crop(position)
        forward()
        torch.cuda.synchronize(input_device)

    crop(position)
    eager_outputs_first = forward()
    torch.cuda.synchronize(input_device)
    eager_logits_first = (
        eager_outputs_first.logits[:, -1, :].detach().clone()
    )
    crop(position)
    eager_outputs_second = forward()
    torch.cuda.synchronize(input_device)
    eager_logits = eager_outputs_second.logits[:, -1, :].detach().clone()

    crop(position)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = forward()
    torch.cuda.synchronize(input_device)
    crop(position)
    graph.replay()
    torch.cuda.synchronize(input_device)
    replay_logits = graph_outputs.logits[:, -1, :].detach().clone()
    graph.replay()
    torch.cuda.synchronize(input_device)
    replay_logits_second = (
        graph_outputs.logits[:, -1, :].detach().clone()
    )

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(input_device)
    wall_start = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    torch.cuda.synchronize(input_device)
    wall_ms = 1000.0 * (time.perf_counter() - wall_start) / iterations

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    stop.record()
    torch.cuda.synchronize(input_device)
    cuda_ms = float(start.elapsed_time(stop)) / iterations
    crop(position)

    replay_max_abs = float((eager_logits - replay_logits).abs().max().item())
    eager_repeat_max_abs = float(
        (eager_logits_first - eager_logits).abs().max().item()
    )
    replay_repeat_max_abs = float(
        (replay_logits - replay_logits_second).abs().max().item()
    )
    eager_top1 = int(eager_logits.argmax(dim=-1).item())
    replay_top1 = int(replay_logits.argmax(dim=-1).item())
    return {
        "position": int(position),
        "warmup": int(warmup),
        "iterations": int(iterations),
        "wall_ms": wall_ms,
        "cuda_ms": cuda_ms,
        "replay_logit_max_abs": replay_max_abs,
        "eager_repeat_logit_max_abs": eager_repeat_max_abs,
        "replay_repeat_logit_max_abs": replay_repeat_max_abs,
        "eager_top1": eager_top1,
        "replay_top1": replay_top1,
        "top1_equal": eager_top1 == replay_top1,
    }


@torch.inference_mode()
def benchmark_growing_qksieve_model_cudagraph(
    model: torch.nn.Module,
    seed_token_id: int,
    past_key_values: PreallocatedDynamicCache,
    input_device: torch.device,
    correctness_steps: int,
    warmup_steps: int,
    timing_steps: int,
    use_qksieve: bool,
) -> dict[str, Any]:
    """Capture autonomous greedy decode over a fixed prefix and growing suffix."""
    prefix_count = int(past_key_values.get_seq_length())
    physical_key_count = int(past_key_values.max_cache_len)
    maximum_steps = physical_key_count - prefix_count
    if maximum_steps <= 0:
        raise RuntimeError("preallocated cache has no graph-decode suffix capacity")
    if min(correctness_steps, warmup_steps, timing_steps) <= 0:
        raise ValueError("growing CUDA Graph step counts must be positive")
    if max(correctness_steps, warmup_steps, timing_steps) > maximum_steps:
        raise ValueError(
            "growing CUDA Graph steps exceed cache capacity: "
            f"{max(correctness_steps, warmup_steps, timing_steps)} > {maximum_steps}"
        )

    past_key_values.enable_graph_decode(prefix_count)
    active_key_count = past_key_values.graph_active_key_count
    static_cache_position = past_key_values.graph_cache_position
    if active_key_count is None or static_cache_position is None:
        raise RuntimeError("graph cache tensors were not initialized")
    graph_config = (
        sparse_attention.configure_qksieve_static_prefix_graph_decode(
            prefix_count,
            physical_key_count,
            active_key_count,
        )
        if use_qksieve
        else {
            "layer_count": len(past_key_values.key_cache),
            "prefix_count": prefix_count,
            "suffix_capacity": physical_key_count - prefix_count,
        }
    )
    static_input_ids = torch.tensor(
        [[int(seed_token_id)]], dtype=torch.long, device=input_device
    )
    model_dtype = next(model.parameters()).dtype
    static_attention_mask_tensor = torch.zeros(
        1,
        1,
        1,
        physical_key_count,
        dtype=model_dtype,
        device=input_device,
    )
    static_attention_mask = {"full_attention": static_attention_mask_tensor}
    mask_positions = torch.arange(
        physical_key_count, dtype=torch.long, device=input_device
    ).view(1, 1, 1, physical_key_count)
    valid_mask_value = torch.zeros((), dtype=model_dtype, device=input_device)
    blocked_mask_value = torch.full(
        (), torch.finfo(model_dtype).min, dtype=model_dtype, device=input_device
    )

    def forward() -> Any:
        if not use_qksieve:
            static_attention_mask_tensor.copy_(
                torch.where(
                    mask_positions <= static_cache_position.view(1, 1, 1, 1),
                    valid_mask_value,
                    blocked_mask_value,
                )
            )
        return lb.model_forward(
            model,
            {
                "input_ids": static_input_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
                "cache_position": static_cache_position,
                "attention_mask": static_attention_mask,
            },
        )

    def eager_autonomous_step() -> Any:
        outputs = forward()
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        static_input_ids.copy_(next_token)
        static_cache_position.add_(1)
        return outputs

    def reset() -> None:
        past_key_values.set_graph_position(prefix_count)
        static_input_ids.fill_(int(seed_token_id))

    # Initialize the enlarged workspaces and extension handles before capture.
    reset()
    for _ in range(2):
        eager_autonomous_step()
    torch.cuda.synchronize(input_device)

    reset()
    eager_logits: list[torch.Tensor] = []
    eager_tokens: list[int] = []
    for _ in range(correctness_steps):
        outputs = eager_autonomous_step()
        logits = outputs.logits[:, -1, :].detach().clone()
        eager_logits.append(logits)
        eager_tokens.append(int(logits.argmax(dim=-1).item()))
    torch.cuda.synchronize(input_device)

    reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = forward()
        graph_next_token = graph_outputs.logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        static_input_ids.copy_(graph_next_token)
        static_cache_position.add_(1)
    torch.cuda.synchronize(input_device)

    reset()
    graph_logits: list[torch.Tensor] = []
    graph_tokens: list[int] = []
    for _ in range(correctness_steps):
        graph.replay()
        torch.cuda.synchronize(input_device)
        logits = graph_outputs.logits[:, -1, :].detach().clone()
        graph_logits.append(logits)
        graph_tokens.append(int(logits.argmax(dim=-1).item()))
    logit_max_abs_by_step = [
        float((eager - replay).abs().max().item())
        for eager, replay in zip(eager_logits, graph_logits)
    ]

    reset()
    for _ in range(warmup_steps):
        graph.replay()
    torch.cuda.synchronize(input_device)

    reset()
    torch.cuda.synchronize(input_device)
    graph_nsys_capture = os.environ.get(
        "QKSIEVE_GRAPH_NSYS_CAPTURE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if graph_nsys_capture:
        torch.cuda.cudart().cudaProfilerStart()
    wall_start = time.perf_counter()
    for _ in range(timing_steps):
        graph.replay()
    torch.cuda.synchronize(input_device)
    wall_ms = 1000.0 * (time.perf_counter() - wall_start) / timing_steps
    if graph_nsys_capture:
        torch.cuda.cudart().cudaProfilerStop()

    reset()
    torch.cuda.synchronize(input_device)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timing_steps):
        graph.replay()
    stop.record()
    torch.cuda.synchronize(input_device)
    cuda_ms = float(start.elapsed_time(stop)) / timing_steps

    result = {
        "prefix_count": prefix_count,
        "physical_key_count": physical_key_count,
        "maximum_suffix_steps": maximum_steps,
        "correctness_steps": correctness_steps,
        "warmup_steps": warmup_steps,
        "timing_steps": timing_steps,
        "wall_ms_per_token": wall_ms,
        "cuda_ms_per_token": cuda_ms,
        "eager_tokens": eager_tokens,
        "graph_tokens": graph_tokens,
        "tokens_equal": eager_tokens == graph_tokens,
        "logit_max_abs_by_step": logit_max_abs_by_step,
        "logit_max_abs": max(logit_max_abs_by_step),
        "graph_config": graph_config,
        "method": "qksieve" if use_qksieve else "full_attention",
    }
    past_key_values.disable_graph_decode(prefix_count)
    return result


def sparse_context(args: argparse.Namespace, method: str):
    if method == "full_attention":
        set_attention_implementation(args.model, "sdpa")
        return nullcontext()
    set_attention_implementation(args.model, "eager")
    if method == "exact_top2":
        return head_top_fraction_mode(0.02)
    if method == "exact_top_fraction":
        return head_top_fraction_mode(args.exact_fraction)
    if method == "exact_top_k":
        return head_top_k_mode(args.exact_tokens)
    sample_fraction = min(1.0, args.sample_count / args.history_tokens)
    return head_qabs_sampled_mass_mode(
        mass_threshold=1.0e-6,
        budget_fractions=(args.direct_fraction,),
        sample_fraction=sample_fraction,
        qabs_dim_count=8,
        candidate_fraction=args.direct_fraction,
        use_cuda_kernels=True,
        skip_candidate_rerank=False,
        qabs_int2_onthefly=False,
        score_mode=args.direct_score_mode,
        projection_dim=args.projection_dim,
        gqa_candidate_mode="independent",
        direct_min_tokens=args.direct_min_tokens,
        direct_max_tokens=args.direct_max_tokens,
        direct_candidate_multiplier=args.candidate_overfetch,
        sampled_quantile_sample_count=args.sample_count,
        protected_recent_tokens=args.protect_recent_tokens,
        qk_metric_query_shrinkage=args.qk_metric_query_shrinkage,
    )


def materialize_packed_qmse_template(
    model: torch.nn.Module,
    template: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Place model-level QKSieve templates beside their decoder layers."""
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
def evaluate_method(
    args: argparse.Namespace,
    tokenizer: Any,
    history: list[int],
    target_ids: list[int],
    method: str,
    input_device: torch.device,
    reference_logits: list[torch.Tensor] | None = None,
    capture_logits: bool = False,
    shared_prefill: tuple[
        Any,
        torch.Tensor,
        float,
        dict[int, torch.Tensor],
    ]
    | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[torch.Tensor] | None,
]:
    peak_memory_allocated: list[int] = []
    peak_memory_reserved: list[int] = []
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)
            torch.cuda.reset_peak_memory_stats(device_index)
    packed_template = None
    template_prepare_seconds = 0.0
    if method == "direct_countcap" and args.packed_qmse_template_in is not None:
        cache_key = str(args.packed_qmse_template_in.resolve())
        cached_template = getattr(
            args,
            "_packed_qmse_resident_template",
            None,
        )
        cached_key = getattr(
            args,
            "_packed_qmse_resident_template_key",
            None,
        )
        if cached_template is None or cached_key != cache_key:
            prepare_start = time.perf_counter()
            loaded_template = torch.load(
                args.packed_qmse_template_in,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(loaded_template, dict):
                raise TypeError(
                    "packed qMSE template must be a layer dictionary"
                )
            packed_template = materialize_packed_qmse_template(
                args.model,
                loaded_template,
            )
            template_prepare_seconds = time.perf_counter() - prepare_start
            args._packed_qmse_resident_template = packed_template
            args._packed_qmse_resident_template_key = cache_key
        else:
            packed_template = cached_template
    capture_prefill_queries = (
        method == "direct_countcap"
        and args.direct_score_mode in PACKED_PREFILL_QUERY_SCORE_MODES
        and (
            args.packed_qmse_template_in is None
            or args.direct_score_mode
            in PACKED_FREQUENCY_PREFILL_QUERY_SCORE_MODES
        )
    )
    if shared_prefill is None:
        prefill_context = (
            prefill_query_tail_mode(8)
            if capture_prefill_queries
            else nullcontext({})
        )
        with prefill_context as prefill_queries:
            cache, previous_logits, dense_seconds = dense_prompt(
                args.model,
                tokenizer,
                history,
                input_device,
                args.prefill_chunk_tokens,
                args.cache_mode,
                args.preallocated_cache_min_tokens,
                len(target_ids),
            )
    else:
        cache, previous_logits, dense_seconds, prefill_queries = shared_prefill
        get_seq_length = getattr(cache, "get_seq_length", None)
        if not callable(get_seq_length):
            raise TypeError("shared prefill cache must expose get_seq_length()")
        cache_tokens = int(get_seq_length())
        if cache_tokens != len(history):
            raise ValueError(
                "shared prefill cache length mismatch: "
                f"{cache_tokens} != {len(history)}"
            )
        if capture_prefill_queries and not prefill_queries:
            raise RuntimeError(
                "frequency-aware shared prefill requires prompt-tail Queries"
            )
    nll_values = [token_nll(previous_logits, int(target_ids[0]))]
    first_row = {
        "target_index": 0,
        "token_id": int(target_ids[0]),
        "nll": nll_values[0],
        "sparse_forward_seconds": 0.0,
    }
    captured_logits = [] if capture_logits else None
    if captured_logits is not None:
        captured_logits.append(previous_logits[0].detach().to("cpu", torch.float16))
    if reference_logits is not None:
        first_row.update(
            logit_stability_metrics(
                reference_logits[0],
                previous_logits[0],
                int(target_ids[0]),
            )
        )
    token_rows = [first_row]
    sparse_seconds = 0.0
    cuda_profiler_range = (
        os.environ.get("QKSIEVE_CUDA_PROFILER_RANGE", "").strip().lower()
        in {"1", "true", "yes"}
    )
    cuda_profiler_method = os.environ.get(
        "QKSIEVE_CUDA_PROFILER_METHOD", "direct_countcap"
    ).strip()
    if cuda_profiler_method not in SUPPORTED_METHODS:
        raise ValueError(f"invalid CUDA profiler method: {cuda_profiler_method}")
    cuda_profiler_start_step = int(
        os.environ.get("QKSIEVE_CUDA_PROFILER_START_STEP", "2")
    )
    cuda_profiler_steps = int(
        os.environ.get("QKSIEVE_CUDA_PROFILER_STEPS", "4")
    )
    if cuda_profiler_start_step < 0 or cuda_profiler_steps <= 0:
        raise ValueError("invalid CUDA profiler range")
    cuda_profiler_active = False
    cuda_profiler_completed = False
    cpu_profiler_range = (
        os.environ.get("QKSIEVE_CPROFILE_RANGE", "").strip().lower()
        in {"1", "true", "yes"}
    )
    cpu_profiler_method = os.environ.get(
        "QKSIEVE_CPROFILE_METHOD", "direct_countcap"
    ).strip()
    if cpu_profiler_method not in SUPPORTED_METHODS:
        raise ValueError(f"invalid CPU profiler method: {cpu_profiler_method}")
    cpu_profiler_start_step = int(
        os.environ.get("QKSIEVE_CPROFILE_START_STEP", "2")
    )
    cpu_profiler_steps = int(
        os.environ.get("QKSIEVE_CPROFILE_STEPS", "1")
    )
    if cpu_profiler_start_step < 0 or cpu_profiler_steps <= 0:
        raise ValueError("invalid CPU profiler range")
    cpu_profiler = cProfile.Profile()
    cpu_profiler_active = False
    cpu_profiler_completed = False
    configured_counts = []
    configured_ratios = []
    actual_count_means = []
    actual_count_min = None
    actual_count_max = None
    pca_basis_source_history_count = 0
    pca_basis_sample_count = 0
    packed_index_ratio_of_full_kv = 0.0
    packed_value_sketch_ratio_of_full_kv = 0.0
    packed_value_residual_risk_ratio_of_full_kv = 0.0
    packed_block_conditional_residual_ratio_of_full_kv = 0.0
    packed_total_auxiliary_ratio_of_full_kv = 0.0
    packed_value_sketch_tail_alpha = 0.0
    conditional_query_gain_mean = 0.0
    conditional_query_gain_min = 0.0
    conditional_query_gain_max = 0.0
    conditional_query_error_reduction_mean = 0.0
    packed_index_rebuild_count = 0
    packed_sample_count = 0
    packed_allocation_frozen = False
    packed_prefill_query_tokens = 0
    packed_transform = ""
    packed_key_centered = False
    packed_sharedtail = False
    packed_mean_bits_by_band: list[float] = []
    packed_active_fraction_by_band: list[float] = []
    frequency_hard_skip_pool_fraction_mean = 0.0
    frequency_hard_skip_state_count = 0
    exact_selection_diagnostic_calls = 0
    exact_topk_recall_mean = 0.0
    selected_attention_mass_mean = 0.0
    oracle_topk_attention_mass_mean = 0.0
    selected_to_oracle_mass_ratio_mean = 0.0
    exact_boundary_gap_mean = 0.0
    exact_boundary_gap_over_score_std_mean = 0.0
    selected_floor_regret_over_score_std_mean = 0.0
    packed_capacity_fraction_mean = 0.0
    packed_attention_split_counts: list[int] = []
    candidate_overflow_rates = []
    candidate_overflow_count_max = 0
    value_refinement_rates = []
    value_refinement_count = 0
    value_refinement_head_count = 0
    packed_mass_ladder_target = 0.0
    packed_mass_ladder_measured_mass_mean = 0.0
    packed_mass_ladder_measured_mass_min = 0.0
    packed_mass_ladder_insufficient_head_fraction = 0.0
    packed_mass_ladder_selected_rung_mean = 0.0
    packed_mass_ladder_selected_rung_max = 0
    parallel_qk_prebuild_stats: dict[str, float | int] = {}
    parallel_value_factor_stats: dict[str, float | int] = {}
    resident_value_install_stats: dict[str, float | int] = {}
    parallel_index_prebuild_stats: dict[str, float | int] = {}
    model_cudagraph_probe: dict[str, Any] = {}
    growing_model_cudagraph_probe: dict[str, Any] = {}

    with sparse_context(args, method):
        binarypc_projections = getattr(args, "binarypc_projections", None)
        if method == "direct_countcap" and binarypc_projections is not None:
            install_active_binarypc_projections(binarypc_projections)
        if packed_template is not None:
            install_active_packed_qmse_templates(packed_template)
        if capture_prefill_queries:
            seed_packed_qmse_prefill_queries(prefill_queries)
        if method == "direct_countcap":
            resident_value_install_stats = (
                sparse_attention.install_resident_value_sketch_cache(cache)
            )
            sparse_seconds += float(
                resident_value_install_stats["seconds"]
            )
        parallel_qk_workers = int(
            os.environ.get("QKSIEVE_PARALLEL_QK_WORKERS", "0")
        )
        if (
            method == "direct_countcap"
            and capture_prefill_queries
            and parallel_qk_workers > 0
        ):
            parallel_qk_prebuild_stats = (
                sparse_attention.precompute_active_packed_qmse_qk_factors(
                    cache,
                    max_workers=parallel_qk_workers,
                )
            )
            sparse_seconds += float(
                parallel_qk_prebuild_stats["total_seconds"]
            )
        parallel_value_workers = int(
            os.environ.get("QKSIEVE_PARALLEL_VALUE_WORKERS", "0")
        )
        if (
            method == "direct_countcap"
            and parallel_value_workers > 0
            and int(resident_value_install_stats.get("layers", 0)) == 0
        ):
            parallel_value_factor_stats = (
                sparse_attention.precompute_active_value_sketch_factors(
                    cache,
                    args.model,
                    max_workers=parallel_value_workers,
                )
            )
            sparse_seconds += float(
                parallel_value_factor_stats["total_seconds"]
            )
        parallel_index_workers = int(
            os.environ.get("QKSIEVE_PARALLEL_INDEX_WORKERS", "0")
        )
        if (
            method == "direct_countcap"
            and capture_prefill_queries
            and parallel_index_workers > 0
        ):
            parallel_index_prebuild_stats = (
                sparse_attention.prebuild_active_packed_qmse_indices(
                    cache,
                    max_workers=parallel_index_workers,
                )
            )
            sparse_seconds += float(
                parallel_index_prebuild_stats["total_seconds"]
            )
        for offset, input_id in enumerate(target_ids[:-1]):
            if (
                cuda_profiler_range
                and method == cuda_profiler_method
                and offset == cuda_profiler_start_step
            ):
                for device_index in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(device_index)
                torch.cuda.cudart().cudaProfilerStart()
                cuda_profiler_active = True
            if (
                cpu_profiler_range
                and method == cpu_profiler_method
                and offset == cpu_profiler_start_step
            ):
                cpu_profiler.enable()
                cpu_profiler_active = True
            collect_exact_oracle_stats = method == "exact_top_k"
            cache, previous_logits, seconds, attention_features = run_one_token(
                args.model,
                int(input_id),
                cache,
                args.history_tokens + offset,
                input_device,
                collect_attention_stats=collect_exact_oracle_stats,
            )
            if (
                cpu_profiler_active
                and offset + 1
                == cpu_profiler_start_step + cpu_profiler_steps
            ):
                cpu_profiler.disable()
                cpu_profiler_active = False
                cpu_profiler_completed = True
                profile_path_text = os.environ.get(
                    "QKSIEVE_CPROFILE_OUT",
                    str(Path.cwd() / f"cprofile_{method}.prof"),
                ).format(method=method)
                profile_path = Path(profile_path_text)
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                cpu_profiler.dump_stats(profile_path)
            if (
                cuda_profiler_active
                and offset + 1
                == cuda_profiler_start_step + cuda_profiler_steps
            ):
                for device_index in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(device_index)
                torch.cuda.cudart().cudaProfilerStop()
                cuda_profiler_active = False
                cuda_profiler_completed = True
            label = int(target_ids[offset + 1])
            nll = token_nll(previous_logits, label)
            nll_values.append(nll)
            sparse_seconds += seconds
            token_rows.append(
                {
                    "target_index": offset + 1,
                    "token_id": label,
                    "nll": nll,
                    "sparse_forward_seconds": seconds,
                }
            )
            if attention_features:
                token_rows[-1].update(attention_features)
            crossing_stats = active_crossing_stats()
            if crossing_stats is not None:
                token_rows[-1].update(
                    {
                        "crossing_rescue_tokens_mean": crossing_stats[0],
                        "crossing_rescue_tokens_max": crossing_stats[1],
                        "crossing_expected_count_mean": crossing_stats[2],
                    }
                )
            stage_timings = (
                sparse_attention.consume_speed_stage_timings()
            )
            if stage_timings:
                token_rows[-1]["speed_stage_ms"] = stage_timings
            if captured_logits is not None:
                captured_logits.append(
                    previous_logits[0].detach().to("cpu", torch.float16)
                )
            if reference_logits is not None:
                token_rows[-1].update(
                    logit_stability_metrics(
                        reference_logits[offset + 1],
                        previous_logits[0],
                        label,
                    )
                )
            if method == "direct_countcap":
                history_count = args.history_tokens + offset
                count = direct_countcap_target_count(
                    history_count,
                    args.direct_fraction,
                    args.direct_min_tokens,
                    args.direct_max_tokens,
                )
                configured_counts.append(count)
                configured_ratios.append(count / history_count)
                actual_stats = active_candidate_count_stats()
                if actual_stats is not None:
                    actual_mean, actual_min, actual_max = actual_stats
                    actual_count_means.append(actual_mean)
                    actual_count_min = (
                        actual_min
                        if actual_count_min is None
                        else min(actual_count_min, actual_min)
                    )
                    actual_count_max = (
                        actual_max
                        if actual_count_max is None
                        else max(actual_count_max, actual_max)
                    )
                    token_rows[-1].update(
                        actual_attention_tokens_mean=actual_mean,
                        actual_attention_tokens_min=actual_min,
                        actual_attention_tokens_max=actual_max,
                    )
                if os.environ.get(
                    "QKSIEVE_RECORD_LAYER_CANDIDATE_STATS", "0"
                ) == "1":
                    token_rows[-1]["candidate_counts_by_layer"] = (
                        active_candidate_count_stats_by_layer()
                    )
                overflow_stats = active_candidate_overflow_stats()
                if overflow_stats is not None:
                    overflow_rate, overflow_count = overflow_stats
                    candidate_overflow_rates.append(overflow_rate)
                    candidate_overflow_count_max = max(
                        candidate_overflow_count_max,
                        overflow_count,
                    )
                    token_rows[-1].update(
                        candidate_overflow_rate=overflow_rate,
                        candidate_overflow_count=overflow_count,
                    )
                refinement_stats = active_value_refinement_stats()
                if refinement_stats is not None:
                    (
                        refinement_rate,
                        refinement_count,
                        refinement_heads,
                    ) = refinement_stats
                    value_refinement_rates.append(refinement_rate)
                    value_refinement_count += refinement_count
                    value_refinement_head_count += refinement_heads
                    token_rows[-1].update(
                        value_refinement_rate=refinement_rate,
                        value_refinement_count=refinement_count,
                        value_refinement_head_count=refinement_heads,
                    )
        graph_probe_enabled = os.environ.get(
            "QKSIEVE_MODEL_CUDAGRAPH_PROBE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        graph_probe_method = os.environ.get(
            "QKSIEVE_MODEL_CUDAGRAPH_METHOD", "direct_countcap"
        ).strip()
        if graph_probe_enabled and (
            method == graph_probe_method or graph_probe_method == "all"
        ):
            graph_position = int(cache.get_seq_length())
            model_cudagraph_probe = benchmark_fixed_position_model_cudagraph(
                args.model,
                int(target_ids[-1]),
                cache,
                graph_position,
                input_device,
                warmup=int(
                    os.environ.get("QKSIEVE_MODEL_CUDAGRAPH_WARMUP", "5")
                ),
                iterations=int(
                    os.environ.get("QKSIEVE_MODEL_CUDAGRAPH_ITERATIONS", "50")
                ),
            )
        growing_graph_probe_enabled = os.environ.get(
            "QKSIEVE_MODEL_GROWING_CUDAGRAPH_PROBE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        growing_graph_probe_method = os.environ.get(
            "QKSIEVE_MODEL_GROWING_CUDAGRAPH_METHOD", "all"
        ).strip()
        if growing_graph_probe_enabled and method in {
            "full_attention",
            "direct_countcap",
        } and (
            growing_graph_probe_method == "all"
            or method == growing_graph_probe_method
        ):
            if not isinstance(cache, PreallocatedDynamicCache):
                raise TypeError(
                    "growing CUDA Graph decode requires --cache_mode preallocated"
                )
            growing_model_cudagraph_probe = (
                benchmark_growing_qksieve_model_cudagraph(
                    args.model,
                    int(target_ids[-1]),
                    cache,
                    input_device,
                    correctness_steps=int(
                        os.environ.get(
                            "QKSIEVE_MODEL_GROWING_GRAPH_CORRECTNESS_STEPS", "8"
                        )
                    ),
                    warmup_steps=int(
                        os.environ.get(
                            "QKSIEVE_MODEL_GROWING_GRAPH_WARMUP_STEPS", "5"
                        )
                    ),
                    timing_steps=int(
                        os.environ.get(
                            "QKSIEVE_MODEL_GROWING_GRAPH_TIMING_STEPS", "50"
                        )
                    ),
                    use_qksieve=method == "direct_countcap",
                )
            )
        if method == "direct_countcap":
            states = sparse_attention._ACTIVE_QABS_PCA_STATES or {}
            source_counts = [
                int(state["pca_basis_source_history_count"])
                for state in states.values()
                if "pca_basis_source_history_count" in state
            ]
            sample_counts = [
                int(state["pca_basis_sample_count"])
                for state in states.values()
                if "pca_basis_sample_count" in state
            ]
            if source_counts:
                pca_basis_source_history_count = max(source_counts)
            if sample_counts:
                pca_basis_sample_count = max(sample_counts)
            packed_bits = [
                float(state["hierarchical_logical_bits_per_token"])
                for state in states.values()
                if "hierarchical_logical_bits_per_token" in state
            ]
            if packed_bits:
                packed_index_ratio_of_full_kv = (
                    sum(packed_bits) / len(packed_bits) / 4096.0
                )
            value_sketch_bits = [
                float(
                    state.get(
                        "qksieve_value_sketch_logical_bits_per_token",
                        state.get(
                            "qksieve_conditional_tail_logical_bits_per_token",
                            0.0,
                        ),
                    )
                )
                + float(
                    state.get(
                        "qksieve_value_sketch_scale_metadata_bits_per_token",
                        0.0,
                    )
                )
                + float(
                    state.get(
                        "qksieve_conditional_value_residual_metadata_bits_per_token",
                        0.0,
                    )
                )
                for state in states.values()
                if "qksieve_value_sketch_logical_bits_per_token" in state
                or "qksieve_conditional_tail_logical_bits_per_token" in state
            ]
            if value_sketch_bits:
                packed_value_sketch_ratio_of_full_kv = (
                    sum(value_sketch_bits) / len(value_sketch_bits) / 4096.0
                )
            value_residual_risk_bits = [
                float(
                    state.get(
                        "qksieve_value_residual_risk_logical_bits_per_token",
                        0.0,
                    )
                )
                + float(
                    state.get(
                        "qksieve_value_residual_risk_metadata_bits_per_token",
                        0.0,
                    )
                )
                for state in states.values()
                if "qksieve_value_residual_risk_logical_bits_per_token" in state
            ]
            if value_residual_risk_bits:
                packed_value_residual_risk_ratio_of_full_kv = (
                    sum(value_residual_risk_bits)
                    / len(value_residual_risk_bits)
                    / 4096.0
                )
            block_conditional_bits = [
                float(
                    state[
                        "qksieve_block_conditional_residual_bits_per_token"
                    ]
                )
                for state in states.values()
                if "qksieve_block_conditional_residual_bits_per_token"
                in state
            ]
            if block_conditional_bits:
                packed_block_conditional_residual_ratio_of_full_kv = (
                    sum(block_conditional_bits)
                    / len(block_conditional_bits)
                    / 4096.0
                )
            value_sketch_tail_alphas = [
                float(state["packed_qmse_value_sketch_tail_alpha"])
                for state in states.values()
                if int(state.get("packed_qmse_value_sketch_rank", 0)) > 0
                and "packed_qmse_value_sketch_tail_alpha" in state
            ]
            if value_sketch_tail_alphas:
                packed_value_sketch_tail_alpha = (
                    sum(value_sketch_tail_alphas)
                    / len(value_sketch_tail_alphas)
                )
            conditional_query_gains = [
                float(
                    state[
                        "qksieve_conditional_value_residual_query_gain"
                    ]
                )
                for state in states.values()
                if "qksieve_conditional_value_residual_query_gain" in state
            ]
            if conditional_query_gains:
                conditional_query_gain_mean = sum(
                    conditional_query_gains
                ) / len(conditional_query_gains)
                conditional_query_gain_min = min(conditional_query_gains)
                conditional_query_gain_max = max(conditional_query_gains)
            conditional_query_reductions = [
                float(
                    state[
                        "qksieve_conditional_value_residual_query_error_reduction"
                    ]
                )
                for state in states.values()
                if "qksieve_conditional_value_residual_query_error_reduction"
                in state
            ]
            finite_query_reductions = [
                value
                for value in conditional_query_reductions
                if math.isfinite(value)
            ]
            if finite_query_reductions:
                conditional_query_error_reduction_mean = sum(
                    finite_query_reductions
                ) / len(finite_query_reductions)
            packed_total_auxiliary_ratio_of_full_kv = (
                packed_index_ratio_of_full_kv
                + packed_value_sketch_ratio_of_full_kv
                + packed_value_residual_risk_ratio_of_full_kv
                + packed_block_conditional_residual_ratio_of_full_kv
            )
            rebuild_counts = [
                int(state["packed_qmse_rebuild_count"])
                for state in states.values()
                if "packed_qmse_rebuild_count" in state
            ]
            if rebuild_counts:
                packed_index_rebuild_count = max(rebuild_counts)
            packed_samples = [
                int(state["packed_qmse_sample_count"])
                for state in states.values()
                if "packed_qmse_sample_count" in state
            ]
            if packed_samples:
                packed_sample_count = max(packed_samples)
            packed_capacity_fractions = [
                float(state["packed_qmse_capacity_fraction"])
                for state in states.values()
                if "packed_qmse_capacity_fraction" in state
            ]
            if packed_capacity_fractions:
                packed_capacity_fraction_mean = (
                    sum(packed_capacity_fractions)
                    / len(packed_capacity_fractions)
                )
            packed_attention_split_counts = sorted(
                {
                    int(state["packed_qmse_attention_split_count"])
                    for state in states.values()
                    if "packed_qmse_attention_split_count" in state
                }
            )
            packed_states = [
                state
                for state in states.values()
                if "packed_qmse_allocation_frozen" in state
            ]
            packed_allocation_frozen = bool(packed_states) and all(
                bool(state["packed_qmse_allocation_frozen"])
                for state in packed_states
            )
            prefill_query_counts = [
                int(state["packed_qmse_prefill_query_tokens"])
                for state in packed_states
                if "packed_qmse_prefill_query_tokens" in state
            ]
            if prefill_query_counts:
                packed_prefill_query_tokens = max(prefill_query_counts)
            transforms = {
                str(state["packed_qmse_transform"])
                for state in packed_states
                if "packed_qmse_transform" in state
            }
            if len(transforms) == 1:
                packed_transform = transforms.pop()
            key_centering_flags = {
                bool(state.get("packed_qmse_key_centered", False))
                for state in packed_states
            }
            if len(key_centering_flags) == 1:
                packed_key_centered = key_centering_flags.pop()
            sharedtail_flags = {
                bool(state.get("packed_qmse_sharedtail", False))
                for state in packed_states
            }
            if len(sharedtail_flags) == 1:
                packed_sharedtail = sharedtail_flags.pop()
            allocations = [
                state["packed_qmse_allocation"].float()
                for state in packed_states
                if isinstance(
                    state.get("packed_qmse_allocation"),
                    torch.Tensor,
                )
            ]
            if allocations:
                flat_allocations = torch.cat(
                    [
                        allocation.reshape(-1, 8).cpu()
                        for allocation in allocations
                    ],
                    dim=0,
                )
                packed_mean_bits_by_band = [
                    float(value)
                    for value in flat_allocations.mean(dim=0).tolist()
                ]
                packed_active_fraction_by_band = [
                    float(value)
                    for value in (
                        flat_allocations > 0
                    ).float().mean(dim=0).tolist()
                ]
            hard_skip_pool_fractions = [
                float(state["frequency_hard_skip_pool_fraction_sum"])
                / int(state["frequency_hard_skip_pool_fraction_calls"])
                for state in states.values()
                if int(
                    state.get(
                        "frequency_hard_skip_pool_fraction_calls", 0
                    )
                )
                > 0
            ]
            if hard_skip_pool_fractions:
                frequency_hard_skip_pool_fraction_mean = (
                    sum(hard_skip_pool_fractions)
                    / len(hard_skip_pool_fractions)
                )
                frequency_hard_skip_state_count = len(
                    hard_skip_pool_fractions
                )
            exact_diagnostic_states = [
                state
                for state in states.values()
                if int(state.get("exact_selection_diagnostic_calls", 0)) > 0
            ]
            if exact_diagnostic_states:
                exact_selection_diagnostic_calls = sum(
                    int(state["exact_selection_diagnostic_calls"])
                    for state in exact_diagnostic_states
                )

                def diagnostic_mean(name: str) -> float:
                    per_layer = [
                        float(state[f"{name}_sum"])
                        / int(state["exact_selection_diagnostic_calls"])
                        for state in exact_diagnostic_states
                    ]
                    return sum(per_layer) / len(per_layer)

                exact_topk_recall_mean = diagnostic_mean(
                    "exact_topk_recall"
                )
                selected_attention_mass_mean = diagnostic_mean(
                    "selected_attention_mass"
                )
                oracle_topk_attention_mass_mean = diagnostic_mean(
                    "oracle_topk_attention_mass"
                )
                selected_to_oracle_mass_ratio_mean = diagnostic_mean(
                    "selected_to_oracle_mass_ratio"
                )
                exact_boundary_gap_mean = diagnostic_mean(
                    "exact_boundary_gap"
                )
                exact_boundary_gap_over_score_std_mean = diagnostic_mean(
                    "exact_boundary_gap_over_score_std"
                )
                selected_floor_regret_over_score_std_mean = diagnostic_mean(
                    "selected_floor_regret_over_score_std"
                )
            mass_ladder_states = [
                state
                for state in states.values()
                if float(state.get("packed_qmse_mass_ladder_target", 0.0))
                > 0.0
                and isinstance(
                    state.get("packed_qmse_mass_ladder_mass"),
                    torch.Tensor,
                )
                and isinstance(
                    state.get("packed_qmse_mass_ladder_rungs"),
                    torch.Tensor,
                )
            ]
            if mass_ladder_states:
                packed_mass_ladder_target = sum(
                    float(state["packed_qmse_mass_ladder_target"])
                    for state in mass_ladder_states
                ) / len(mass_ladder_states)
                measured_mass = torch.cat(
                    [
                        state["packed_qmse_mass_ladder_mass"]
                        .detach()
                        .float()
                        .reshape(-1)
                        .cpu()
                        for state in mass_ladder_states
                    ]
                )
                selected_rungs = torch.cat(
                    [
                        state["packed_qmse_mass_ladder_rungs"]
                        .detach()
                        .long()
                        .reshape(-1)
                        .cpu()
                        for state in mass_ladder_states
                    ]
                )
                packed_mass_ladder_measured_mass_mean = float(
                    measured_mass.mean().item()
                )
                packed_mass_ladder_measured_mass_min = float(
                    measured_mass.min().item()
                )
                packed_mass_ladder_insufficient_head_fraction = float(
                    (measured_mass < packed_mass_ladder_target)
                    .float()
                    .mean()
                    .item()
                )
                packed_mass_ladder_selected_rung_mean = float(
                    selected_rungs.float().mean().item()
                )
                packed_mass_ladder_selected_rung_max = int(
                    selected_rungs.max().item()
                )
        if (
            method == "direct_countcap"
            and args.packed_qmse_template_out is not None
        ):
            exported = export_active_packed_qmse_templates()
            if not exported:
                raise RuntimeError(
                    "no packed qMSE layer templates were available"
                )
            args.packed_qmse_template_out.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            torch.save(exported, args.packed_qmse_template_out)

    mean_nll = sum(nll_values) / len(nll_values)
    if method == "full_attention":
        configured_token_count = args.history_tokens
        configured_ratio = 1.0
    elif method == "exact_top2":
        configured_token_count = math.ceil(0.02 * args.history_tokens)
        configured_ratio = 0.02
    elif method == "exact_top_fraction":
        configured_token_count = math.ceil(
            args.exact_fraction * args.history_tokens
        )
        configured_ratio = args.exact_fraction
    elif method == "exact_top_k":
        configured_token_count = min(
            args.history_tokens,
            int(args.exact_tokens),
        )
        configured_ratio = configured_token_count / args.history_tokens
    else:
        configured_token_count = sum(configured_counts) / len(configured_counts)
        configured_ratio = sum(configured_ratios) / len(configured_ratios)
    sparse_step_times = [
        float(row["sparse_forward_seconds"])
        for row in token_rows
        if int(row["target_index"]) > 0
    ]
    packed_steady_times = (
        [
            seconds
            for step, seconds in enumerate(sparse_step_times, start=1)
            if step
            not in (
                {1}
                if (
                    packed_prefill_query_tokens > 0
                    or args.packed_qmse_template_in is not None
                )
                else {1, 8}
            )
        ]
        if args.direct_score_mode in PACKED_QMSE_SCORE_MODES
        and method == "direct_countcap"
        else sparse_step_times
    )
    steady_seconds_per_step = (
        sum(packed_steady_times) / len(packed_steady_times)
        if packed_steady_times
        else 0.0
    )
    stage_rows = [
        row["speed_stage_ms"]
        for row in token_rows
        if int(row["target_index"]) > 1
        and isinstance(row.get("speed_stage_ms"), dict)
    ]
    stage_names = sorted(
        {
            name
            for row in stage_rows
            for name in row
        }
    )
    speed_stage_ms_mean = {
        name: sum(float(row.get(name, 0.0)) for row in stage_rows)
        / len(stage_rows)
        for name in stage_names
    } if stage_rows else {}
    crossing_rows = [
        row
        for row in token_rows
        if int(row["target_index"]) > 1
        and "crossing_rescue_tokens_mean" in row
    ]
    crossing_rescue_tokens_mean = (
        sum(float(row["crossing_rescue_tokens_mean"]) for row in crossing_rows)
        / len(crossing_rows)
        if crossing_rows
        else 0.0
    )
    crossing_rescue_tokens_max = (
        max(int(row["crossing_rescue_tokens_max"]) for row in crossing_rows)
        if crossing_rows
        else 0
    )
    crossing_expected_count_mean = (
        sum(float(row["crossing_expected_count_mean"]) for row in crossing_rows)
        / len(crossing_rows)
        if crossing_rows
        else 0.0
    )
    fixed_overhead_seconds = (
        sparse_seconds
        - steady_seconds_per_step * len(sparse_step_times)
    )
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)
            peak_memory_allocated.append(
                int(torch.cuda.max_memory_allocated(device_index))
            )
            peak_memory_reserved.append(
                int(torch.cuda.max_memory_reserved(device_index))
            )
    summary = {
        "method": method,
        "tokens": len(nll_values),
        "nll": mean_nll,
        "ppl": math.exp(min(20.0, mean_nll)),
        "dense_prompt_seconds": dense_seconds,
        "sparse_decode_seconds": sparse_seconds,
        "sparse_seconds_per_step": (
            sparse_seconds / max(1, len(target_ids) - 1)
        ),
        "configured_attention_tokens_mean": configured_token_count,
        "configured_attention_ratio_mean": configured_ratio,
        "actual_attention_tokens_mean": (
            sum(actual_count_means) / len(actual_count_means)
            if actual_count_means
            else configured_token_count
        ),
        "actual_attention_tokens_min": (
            actual_count_min
            if actual_count_min is not None
            else configured_token_count
        ),
        "actual_attention_tokens_max": (
            actual_count_max
            if actual_count_max is not None
            else configured_token_count
        ),
        "projection_dim": (
            128
            if method == "direct_countcap"
            and args.direct_score_mode in PACKED_QMSE_SCORE_MODES
            else args.projection_dim
            if method == "direct_countcap"
            else 0
        ),
        "pca_basis_source_history_count": (
            pca_basis_source_history_count
            if method == "direct_countcap"
            else 0
        ),
        "pca_basis_sample_count": (
            pca_basis_sample_count if method == "direct_countcap" else 0
        ),
        "protected_recent_tokens": (
            args.protect_recent_tokens if method == "direct_countcap" else 0
        ),
        "candidate_overfetch": (
            args.candidate_overfetch if method == "direct_countcap" else 1.0
        ),
        "score_mode": (
            args.direct_score_mode if method == "direct_countcap" else ""
        ),
        "steady_sparse_seconds_per_step": steady_seconds_per_step,
        "fixed_sparse_overhead_seconds": fixed_overhead_seconds,
        "fixed_position_model_cudagraph": model_cudagraph_probe,
        "growing_model_cudagraph": growing_model_cudagraph_probe,
        "cuda_profiler_range_requested": cuda_profiler_range,
        "cuda_profiler_range_completed": cuda_profiler_completed,
        "cuda_profiler_method": cuda_profiler_method,
        "cuda_profiler_start_step": cuda_profiler_start_step,
        "cuda_profiler_steps": cuda_profiler_steps,
        "cpu_profiler_range_requested": cpu_profiler_range,
        "cpu_profiler_range_completed": cpu_profiler_completed,
        "cpu_profiler_method": cpu_profiler_method,
        "cpu_profiler_start_step": cpu_profiler_start_step,
        "cpu_profiler_steps": cpu_profiler_steps,
        "packed_parallel_qk_prebuild": parallel_qk_prebuild_stats,
        "packed_parallel_value_factors": parallel_value_factor_stats,
        "resident_value_sketch_install": resident_value_install_stats,
        "packed_parallel_index_prebuild": parallel_index_prebuild_stats,
        "speed_stage_ms_mean": speed_stage_ms_mean,
        "crossing_rescue_tokens_mean": crossing_rescue_tokens_mean,
        "crossing_rescue_tokens_max": crossing_rescue_tokens_max,
        "crossing_expected_count_mean": crossing_expected_count_mean,
        "packed_index_ratio_of_full_kv": (
            packed_index_ratio_of_full_kv
            if method == "direct_countcap"
            else 0.0
        ),
        "packed_value_sketch_ratio_of_full_kv": (
            packed_value_sketch_ratio_of_full_kv
            if method == "direct_countcap"
            else 0.0
        ),
        "packed_value_residual_risk_ratio_of_full_kv": (
            packed_value_residual_risk_ratio_of_full_kv
            if method == "direct_countcap"
            else 0.0
        ),
        "packed_block_conditional_residual_ratio_of_full_kv": (
            packed_block_conditional_residual_ratio_of_full_kv
            if method == "direct_countcap"
            else 0.0
        ),
        "packed_total_auxiliary_ratio_of_full_kv": (
            packed_total_auxiliary_ratio_of_full_kv
            if method == "direct_countcap"
            else 0.0
        ),
        "packed_value_sketch_tail_alpha": (
            packed_value_sketch_tail_alpha
            if method == "direct_countcap"
            else 0.0
        ),
        "conditional_query_gain_mean": (
            conditional_query_gain_mean if method == "direct_countcap" else 0.0
        ),
        "conditional_query_gain_min": (
            conditional_query_gain_min if method == "direct_countcap" else 0.0
        ),
        "conditional_query_gain_max": (
            conditional_query_gain_max if method == "direct_countcap" else 0.0
        ),
        "conditional_query_error_reduction_mean": (
            conditional_query_error_reduction_mean
            if method == "direct_countcap"
            else 0.0
        ),
        "value_refinement_rate_mean": (
            sum(value_refinement_rates) / len(value_refinement_rates)
            if method == "direct_countcap" and value_refinement_rates
            else 0.0
        ),
        "value_refinement_count": (
            value_refinement_count if method == "direct_countcap" else 0
        ),
        "value_refinement_head_count": (
            value_refinement_head_count
            if method == "direct_countcap"
            else 0
        ),
        "packed_mass_ladder_target": (
            packed_mass_ladder_target
            if method == "direct_countcap"
            else 0.0
        ),
        "packed_mass_ladder_measured_mass_mean": (
            packed_mass_ladder_measured_mass_mean
            if method == "direct_countcap"
            else 0.0
        ),
        "packed_mass_ladder_measured_mass_min": (
            packed_mass_ladder_measured_mass_min
            if method == "direct_countcap"
            else 0.0
        ),
        "packed_mass_ladder_insufficient_head_fraction": (
            packed_mass_ladder_insufficient_head_fraction
            if method == "direct_countcap"
            else 0.0
        ),
        "packed_mass_ladder_selected_rung_mean": (
            packed_mass_ladder_selected_rung_mean
            if method == "direct_countcap"
            else 0.0
        ),
        "packed_mass_ladder_selected_rung_max": (
            packed_mass_ladder_selected_rung_max
            if method == "direct_countcap"
            else 0
        ),
        "packed_index_rebuild_count": (
            packed_index_rebuild_count
            if method == "direct_countcap"
            else 0
        ),
        "packed_quantile_sample_count": (
            packed_sample_count
            if method == "direct_countcap"
            else 0
        ),
        "packed_capacity_fraction_mean": (
            packed_capacity_fraction_mean
            if method == "direct_countcap"
            else 0.0
        ),
        "packed_attention_split_counts": (
            packed_attention_split_counts
            if method == "direct_countcap"
            else []
        ),
        "packed_allocation_frozen": (
            packed_allocation_frozen
            if method == "direct_countcap"
            else False
        ),
        "packed_prefill_query_tokens": (
            packed_prefill_query_tokens
            if method == "direct_countcap"
            else 0
        ),
        "packed_transform": (
            packed_transform if method == "direct_countcap" else ""
        ),
        "packed_key_centered": (
            packed_key_centered if method == "direct_countcap" else False
        ),
        "packed_sharedtail": (
            packed_sharedtail if method == "direct_countcap" else False
        ),
        "packed_fixed_template_active": (
            args.packed_qmse_template_in is not None
            if method == "direct_countcap"
            else False
        ),
        "packed_template_prepare_seconds": (
            template_prepare_seconds
            if method == "direct_countcap"
            else 0.0
        ),
        "packed_mean_bits_by_band": (
            packed_mean_bits_by_band
            if method == "direct_countcap"
            else []
        ),
        "packed_active_fraction_by_band": (
            packed_active_fraction_by_band
            if method == "direct_countcap"
            else []
        ),
        "frequency_hard_skip_pool_fraction_mean": (
            frequency_hard_skip_pool_fraction_mean
            if method == "direct_countcap"
            else 0.0
        ),
        "frequency_hard_skip_state_count": (
            frequency_hard_skip_state_count
            if method == "direct_countcap"
            else 0
        ),
        "exact_selection_diagnostic_calls": (
            exact_selection_diagnostic_calls
            if method == "direct_countcap"
            else 0
        ),
        "exact_topk_recall_mean": (
            exact_topk_recall_mean if method == "direct_countcap" else 0.0
        ),
        "selected_attention_mass_mean": (
            selected_attention_mass_mean
            if method == "direct_countcap"
            else 0.0
        ),
        "oracle_topk_attention_mass_mean": (
            oracle_topk_attention_mass_mean
            if method == "direct_countcap"
            else 0.0
        ),
        "selected_to_oracle_mass_ratio_mean": (
            selected_to_oracle_mass_ratio_mean
            if method == "direct_countcap"
            else 0.0
        ),
        "exact_boundary_gap_mean": (
            exact_boundary_gap_mean if method == "direct_countcap" else 0.0
        ),
        "exact_boundary_gap_over_score_std_mean": (
            exact_boundary_gap_over_score_std_mean
            if method == "direct_countcap"
            else 0.0
        ),
        "selected_floor_regret_over_score_std_mean": (
            selected_floor_regret_over_score_std_mean
            if method == "direct_countcap"
            else 0.0
        ),
        "candidate_overflow_rate_mean": (
            sum(candidate_overflow_rates) / len(candidate_overflow_rates)
            if candidate_overflow_rates
            else 0.0
        ),
        "candidate_overflow_count_max": candidate_overflow_count_max,
        "history_tokens": args.history_tokens,
        "cache_mode": args.cache_mode,
        "used_preallocated_cache": should_use_preallocated_cache(
            args.cache_mode,
            args.history_tokens,
            args.preallocated_cache_min_tokens,
        ),
        "peak_gpu_allocated_bytes_per_device": peak_memory_allocated,
        "peak_gpu_allocated_bytes_total": sum(peak_memory_allocated),
        "peak_gpu_allocated_bytes_max_device": (
            max(peak_memory_allocated) if peak_memory_allocated else 0
        ),
        "peak_gpu_reserved_bytes_per_device": peak_memory_reserved,
        "peak_gpu_reserved_bytes_total": sum(peak_memory_reserved),
        "peak_gpu_reserved_bytes_max_device": (
            max(peak_memory_reserved) if peak_memory_reserved else 0
        ),
    }
    oracle_stat_names = {
        "retained_mass_mean",
        "retained_mass_min",
        "retained_mass_p10",
        "retained_mass_p25",
        "retained_mass_lt90_fraction",
        "retained_mass_lt95_fraction",
        "retained_mass_lt99_fraction",
        "retained_mass_early_mean",
        "retained_mass_middle_mean",
        "retained_mass_late_mean",
        "top1_attention_mass_mean",
    }
    oracle_stat_names.update(
        name
        for row in token_rows
        for name in row
        if name.startswith("exact_boundary_gap")
        or name.startswith("value_")
    )
    for name in sorted(oracle_stat_names):
        values = [
            float(row[name])
            for row in token_rows
            if name in row
        ]
        if values:
            summary[f"oracle_{name}"] = sum(values) / len(values)
    stability_rows = [
        row for row in token_rows if "top1_agreement" in row
    ]
    if stability_rows:
        summary.update(
            top1_agreement=sum(
                int(row["top1_agreement"]) for row in stability_rows
            )
            / len(stability_rows),
            margin_flip_rate=sum(
                1 - int(row["top1_agreement"]) for row in stability_rows
            )
            / len(stability_rows),
            margin_certificate_rate=sum(
                int(row["margin_certificate_satisfied"])
                for row in stability_rows
            )
            / len(stability_rows),
            kl_full_to_sparse_mean=sum(
                float(row["kl_full_to_sparse"]) for row in stability_rows
            )
            / len(stability_rows),
            js_divergence_mean=sum(
                float(row["js_divergence"]) for row in stability_rows
            )
            / len(stability_rows),
            target_nll_delta_mean=sum(
                float(row["target_nll_delta"]) for row in stability_rows
            )
            / len(stability_rows),
        )
    return summary, token_rows, captured_logits


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    topics = parse_topics(args.topics)
    windows = parse_int_list(args.window_indices)
    if args.history_tokens <= 1 or args.eval_tokens <= 0:
        raise ValueError("history_tokens and eval_tokens must be positive")
    if (
        args.packed_qmse_template_in is not None
        and args.direct_score_mode not in PACKED_QMSE_SCORE_MODES
    ):
        raise ValueError(
            "--packed_qmse_template_in requires a packed qMSE score mode"
        )
    if (
        args.packed_qmse_template_out is not None
        and args.direct_score_mode not in PACKED_QMSE_SCORE_MODES
    ):
        raise ValueError(
            "--packed_qmse_template_out requires a packed qMSE score mode"
        )
    if args.collect_logit_stability and (
        "full_attention" not in methods or methods[0] != "full_attention"
    ):
        raise ValueError(
            "logit stability requires full_attention as the first method"
        )
    direct_countcap_target_count(
        args.history_tokens,
        args.direct_fraction,
        args.direct_min_tokens,
        args.direct_max_tokens,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str),
        encoding="utf-8",
    )

    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(args)
    args.model = model
    bounds = {
        window: window_bounds(
            args.history_tokens,
            window,
            args.window_stride_tokens,
            args.target_anchor_tokens,
        )
        for window in windows
    }
    required_tokens = max(end for _, end in bounds.values()) + args.eval_tokens
    summaries = []
    all_token_rows = []
    for topic in topics:
        stream = encode_evaluation_stream(
            tokenizer,
            topic,
            required_tokens,
            args.dataset_cache_dir,
            args.seed,
        )
        for window in windows:
            start, target_start = bounds[window]
            history = stream[start:target_start]
            target_ids = stream[
                target_start : target_start + args.eval_tokens
            ]
            print(f"[case] topic={topic} window={window}", flush=True)
            case_summaries = []
            full_logits_reference = None
            for method in methods:
                capture_logits = (
                    args.collect_logit_stability
                    and method == "full_attention"
                )
                reference_logits = (
                    full_logits_reference
                    if args.collect_logit_stability
                    and method != "full_attention"
                    else None
                )
                summary, token_rows, captured_logits = evaluate_method(
                    args,
                    tokenizer,
                    history,
                    target_ids,
                    method,
                    input_device,
                    reference_logits=reference_logits,
                    capture_logits=capture_logits,
                )
                if captured_logits is not None:
                    full_logits_reference = captured_logits
                summary.update(
                    topic=topic,
                    window=window,
                    history_start=start,
                    target_start=target_start,
                )
                for row in token_rows:
                    row.update(topic=topic, window=window, method=method)
                case_summaries.append(summary)
                all_token_rows.extend(token_rows)
                print(json.dumps(summary, sort_keys=True), flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            full = next(
                (
                    row
                    for row in case_summaries
                    if row["method"] == "full_attention"
                ),
                None,
            )
            if full is not None:
                for summary in case_summaries:
                    summary["delta_nll_vs_full"] = (
                        summary["nll"] - full["nll"]
                    )
                    summary["ppl_ratio_vs_full"] = (
                        summary["ppl"] / full["ppl"]
                    )
                    summary["quality_retention"] = (
                        full["ppl"] / summary["ppl"]
                    )
            summaries.extend(case_summaries)

    write_csv(args.output_dir / "token_results.csv", all_token_rows)
    write_csv(args.output_dir / "case_summary.csv", summaries)
    (args.output_dir / "case_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
