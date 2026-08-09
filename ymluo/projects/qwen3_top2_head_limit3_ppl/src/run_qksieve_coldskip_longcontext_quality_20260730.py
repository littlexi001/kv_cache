#!/usr/bin/env python
"""Shared-prefill quality check for QKSieve at very long context lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from analyze_qk_product_spectrum_20260803 import (
    analyze_cache_qk_product_spectrum,
)
import run_direct_countcap_denseprompt_ppl_20260725 as direct
from run_head_top2_targeted_ppl_20260714 import (
    direct_countcap_target_count,
    install_llama_head_top_fraction_patch,
    load_model,
    prebuild_resident_qksieve_key_factors,
    prebuild_resident_value_sketch_cache,
    precompute_qksieve_value_metric_grams,
    preload_qksieve_qmse_rate_tables,
    preload_qksieve_runtime_extensions,
    prefill_query_tail_mode,
)


QKSIEVE_KEYMSE_PROXYMASS_SCORE_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_proxymass_unbiased_packed_direct"
)
QKSIEVE_KEYMSE_MEANTAIL_SCORE_MODE = (
    direct.PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_MEANTAIL_SCORE_MODE
)
QKSIEVE_KEYMSE_TILTTAIL16_SCORE_MODE = (
    direct.PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_TILTTAIL16_SCORE_MODE
)
QKSIEVE_KEYMSE_CONDTAIL32_SCORE_MODE = (
    direct.PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_CONDTAIL32_SCORE_MODE
)
QKSIEVE_KEYMSE_CONDTAIL16_SCORE_MODE = (
    direct.PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_CONDTAIL16_SCORE_MODE
)
QKSIEVE_KEYMSE_CONDTAIL8_SCORE_MODE = (
    direct.PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_CONDTAIL8_SCORE_MODE
)

VARIANTS = {
    "binarypc_exactrerank4x_k1280": (
        "binarypc_offline64_exactrerank4x_fulltopk"
    ),
    "binarypc_exactrerank8x_k1280": (
        "binarypc_offline64_exactrerank_fulltopk"
    ),
    "qksieve": direct.PACKED_QMSE_QKMETRIC_FULLTOPK_SCORE_MODE,
    "qksieve_keymse_fulltopk": (
        direct.PACKED_QKBALANCED_AUTOKEY_FULLTOPK_SCORE_MODE
    ),
    "qksieve_deploy": (
        direct.PACKED_QMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE
    ),
    "qksieve_deploy_keymse": (
        direct.PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_sampled_k1280_c32": (
        direct.PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_proxymass_k1280_c32": (
        QKSIEVE_KEYMSE_PROXYMASS_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_meantail_k1280_c32": (
        QKSIEVE_KEYMSE_MEANTAIL_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_meantail_k1280_s512": (
        QKSIEVE_KEYMSE_MEANTAIL_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_tilttail16_k1280_c32": (
        QKSIEVE_KEYMSE_TILTTAIL16_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_condtail32_k1280_c32": (
        QKSIEVE_KEYMSE_CONDTAIL32_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_condtail16_k1280_c32": (
        QKSIEVE_KEYMSE_CONDTAIL16_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_condtail8_k1280_c32": (
        QKSIEVE_KEYMSE_CONDTAIL8_SCORE_MODE
    ),
    "qksieve_fixed410_requestlocal_valuesketch16i4_sampled_k1280": (
        direct
        .PACKED_QMSE_QKMETRIC_FIXED410_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH16_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_valuesketch16i4_massladder90": (
        direct
        .PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_MASSLADDER90_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_valuesketch16i4_massladder95": (
        direct
        .PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_MASSLADDER95_SCORE_MODE
    ),
    "coldskip50": direct.PACKED_QMSE_QKMETRIC_FREQSKIP50_SCORE_MODE,
    "coldskip60": direct.PACKED_QMSE_QKMETRIC_FREQSKIP60_SCORE_MODE,
    "qksieve_keymse_requestlocal_fixedalloc_b12_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED441_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_b15_4421_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED4421_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_b13_4221_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED4221_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_b10_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED440_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_b8_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED420_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_i112_41_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED410_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_i112_41_fulltopk_k2560": (
        direct.PACKED_QMSE_QKMETRIC_FIXED410_FULLTOPK_SCORE_MODE
    ),
    "qksieve_requestlocal_fp16x2_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FP16X2_FULLTOPK_SCORE_MODE
    ),
    "qksieve_requestlocal_fp16x2_fulltopk_k2560": (
        direct.PACKED_QMSE_QKMETRIC_FP16X2_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_i112_211_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED211_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_i96_22_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED220_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xprererank_b8_"
    "fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED420_POST2X_PRERERANK_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xprererank_i112_41_"
    "fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED410_POST2X_PRERERANK_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xprererank_i112_41_"
    "l00to02_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO02_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xprererank_i112_41_"
    "l00to05_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO05_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xprererank_i112_41_"
    "l00to08_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO08_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xprererank_i112_41_"
    "l00to08_fulltopk_k2560": (
        direct.PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO08_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xdualmass_i112_41_"
    "l00to08_fulltopk_k1280": (
        direct
        .PACKED_QMSE_QKMETRIC_FIXED410_POST2X_DUALMASS_L00TO08_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xdualmass_i112_41_"
    "l00to08_fulltopk_k2560": (
        direct
        .PACKED_QMSE_QKMETRIC_FIXED410_POST2X_DUALMASS_L00TO08_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xprererank_i112_41_"
    "l09to17_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L09TO17_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xprererank_i112_41_"
    "l00to17_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO17_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xprererank_i112_41_"
    "l18to26_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L18TO26_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xprererank_i112_41_"
    "l00to26_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L00TO26_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xprererank_i112_41_"
    "l27to35_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L27TO35_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_post2xboundary75prererank_b8_"
    "fulltopk_k1280": (
        direct
        .PACKED_QMSE_QKMETRIC_FIXED420_POST2X_BOUNDARY75_PRERERANK_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_fullprerope_b8_"
    "fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED420_FULL_PREROPE_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_fullprerope_localsink_b8_"
    "fulltopk_k1280": (
        direct
        .PACKED_QMSE_QKMETRIC_FIXED420_FULL_PREROPE_LOCALSINK_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_fullprerope_b10_"
    "fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED440_FULL_PREROPE_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_fullprerope_localsink_b10_"
    "fulltopk_k1280": (
        direct
        .PACKED_QMSE_QKMETRIC_FIXED440_FULL_PREROPE_LOCALSINK_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_fullprerope_b12_"
    "fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED441_FULL_PREROPE_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_prerope32int2_b8_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED420_PREROPE32INT2_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_prerope32int4_b8_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED420_PREROPE32INT4_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_prerope32adaptive_b8_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED420_PREROPE32ADAPTIVE_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_b5_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED400_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_minifloat_b5_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED400_MINIFLOAT_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_b3_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED200_FULLTOPK_SCORE_MODE
    ),
    "qksieve_keymse_requestlocal_fixedalloc_minifloat_b8_fulltopk_k1280": (
        direct.PACKED_QMSE_QKMETRIC_FIXED420_MINIFLOAT_FULLTOPK_SCORE_MODE
    ),
}
EXACT_FP16_QK_ORACLE = "__exact_fp16_qk_oracle__"
for _budget in (256, 492, 984, 1280, 2560, 5120, 10240, 16384):
    VARIANTS[f"exact_qk_oracle_k{_budget}"] = EXACT_FP16_QK_ORACLE
    VARIANTS[f"qksieve_qmse_requestlocal_fulltopk_k{_budget}"] = (
        direct.PACKED_QMSE_QKMETRIC_FULLTOPK_SCORE_MODE
    )
    VARIANTS[
        f"qksieve_qmse_requestlocal_dualmass975_affineres_k{_budget}"
    ] = direct.PACKED_QKSIEVE_QKMSE_DUALMASS975_FULLTOPK_SCORE_MODE
    VARIANTS[
        f"qksieve_qmse_requestlocal_dualmass975_k{_budget}"
    ] = direct.PACKED_QKSIEVE_QMSE_DUALMASS975_NOAFFINE_FULLTOPK_SCORE_MODE
    VARIANTS[
        f"qksieve_qmse_oas_requestlocal_dualmass975_k{_budget}"
    ] = (
        direct
        .PACKED_QKSIEVE_QMSE_OAS_DUALMASS975_NOAFFINE_FULLTOPK_SCORE_MODE
    )
    VARIANTS[
        f"qksieve_qmse_oas_requestlocal_dualmass975_diag_k{_budget}"
    ] = (
        direct
        .PACKED_QKSIEVE_QMSE_OAS_DUALMASS975_DIAGONAL_NOAFFINE_FULLTOPK_SCORE_MODE
    )
    VARIANTS[
        f"qksieve_qmse_oas_requestlocal_valuesketch16_k{_budget}"
    ] = (
        direct
        .PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_FULLTOPK_SCORE_MODE
    )
    VARIANTS[
        f"qksieve_qmse_oas_requestlocal_valuesketch16_sampled_k{_budget}"
    ] = (
        direct
        .PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SAMPLED_SCORE_MODE
    )
    for _sample_count in (512, 1024):
        VARIANTS[
            "qksieve_qmse_oas_requestlocal_valuesketch16_sampled_"
            f"s{_sample_count}_k{_budget}"
        ] = (
            direct
            .PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SAMPLED_SCORE_MODE
        )
    for _sample_count in (512, 1024):
        VARIANTS[
            "qksieve_qmse_oas_requestlocal_valuesketch16_sorted_"
            f"s{_sample_count}_k{_budget}"
        ] = (
            direct
            .PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SORTED_SAMPLED_SCORE_MODE
        )
    for _target_tail_count in (32, 64):
        VARIANTS[
            "qksieve_qmse_oas_requestlocal_valuesketch16_sorted_"
            f"c{_target_tail_count}_k{_budget}"
        ] = (
            direct
            .PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SORTED_SAMPLED_SCORE_MODE
        )
        VARIANTS[
            "qksieve_qmse_oas_requestlocal_valuesketch16_sorted_"
            f"c{_target_tail_count}_novalue_k{_budget}"
        ] = (
            direct
            .PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SORTED_SAMPLED_SCORE_MODE
        )
    VARIANTS[
        f"qksieve_qmse_oas_requestlocal_valuesketch8_k{_budget}"
    ] = (
        direct
        .PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH8_WOMETRIC_FULLTOPK_SCORE_MODE
    )
    VARIANTS[
        f"qksieve_qmse_oas_requestlocal_valuesketch12_k{_budget}"
    ] = (
        direct
        .PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH12_WOMETRIC_FULLTOPK_SCORE_MODE
    )
    VARIANTS[
        f"qksieve_qmse_oas_requestlocal_condres8_k{_budget}"
    ] = direct.PACKED_QKSIEVE_QMSE_OAS_CONDRES8_FULLTOPK_SCORE_MODE
    VARIANTS[
        f"qksieve_qmse_oas_requestlocal_condres8wiener_k{_budget}"
    ] = direct.PACKED_QKSIEVE_QMSE_OAS_CONDRES8_WIENER_FULLTOPK_SCORE_MODE
    VARIANTS[
        f"qksieve_qmse_oas_requestlocal_condres8query_k{_budget}"
    ] = direct.PACKED_QKSIEVE_QMSE_OAS_CONDRES8_QUERY_FULLTOPK_SCORE_MODE
    VARIANTS[
        f"qksieve_qmse_oas_requestlocal_condres8safequery_k{_budget}"
    ] = direct.PACKED_QKSIEVE_QMSE_OAS_CONDRES8_SAFEQUERY_FULLTOPK_SCORE_MODE
    VARIANTS[
        f"qksieve_qmse_oas_requestlocal_prefixrss25e4_k{_budget}"
    ] = direct.PACKED_QKSIEVE_QMSE_OAS_PREFIXRSS25E4_S1_FULLTOPK_SCORE_MODE
    VARIANTS[
        f"qksieve_keymse_requestlocal_dualmass975_k{_budget}"
    ] = direct.PACKED_QKSIEVE_KEYMSE_DUALMASS975_NOAFFINE_FULLTOPK_SCORE_MODE
    VARIANTS[
        f"qksieve_keymse_requestlocal_dualmass975_affineres_k{_budget}"
    ] = direct.PACKED_QKSIEVE_KEYMSE_DUALMASS975_AFFINE_FULLTOPK_SCORE_MODE
    VARIANTS[f"qksieve_qmse_requestlocal_sampled_k{_budget}_c32"] = (
        direct.PACKED_QMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE
    )
    VARIANTS[
        f"qksieve_keymse_requestlocal_fixedalloc_b15_4421_fulltopk_k{_budget}"
    ] = direct.PACKED_QMSE_QKMETRIC_FIXED4421_FULLTOPK_SCORE_MODE
    VARIANTS[
        f"qksieve_keymse_requestlocal_fixedalloc_b13_4221_fulltopk_k{_budget}"
    ] = direct.PACKED_QMSE_QKMETRIC_FIXED4221_FULLTOPK_SCORE_MODE
    VARIANTS[f"qksieve_keymse_requestlocal_fulltopk_k{_budget}"] = (
        direct.PACKED_QKBALANCED_AUTOKEY_FULLTOPK_SCORE_MODE
    )
    for _block_size, _score_mode in (
        (
            32,
            direct.PACKED_QKBALANCED_AUTOKEY_BLOCKMASS32_FULLTOPK_SCORE_MODE,
        ),
        (
            64,
            direct.PACKED_QKBALANCED_AUTOKEY_BLOCKMASS64_FULLTOPK_SCORE_MODE,
        ),
        (
            128,
            direct.PACKED_QKBALANCED_AUTOKEY_BLOCKMASS128_FULLTOPK_SCORE_MODE,
        ),
        (
            256,
            direct.PACKED_QKBALANCED_AUTOKEY_BLOCKMASS256_FULLTOPK_SCORE_MODE,
        ),
        (
            512,
            direct.PACKED_QKBALANCED_AUTOKEY_BLOCKMASS512_FULLTOPK_SCORE_MODE,
        ),
    ):
        VARIANTS[
            f"qksieve_keymse_requestlocal_blockmass{_block_size}_"
            f"fulltopk_k{_budget}"
        ] = _score_mode
    for _name, _score_mode in (
        (
            "blockcalmass64",
            direct.PACKED_QKBALANCED_AUTOKEY_BLOCKCALMASS64_FULLTOPK_SCORE_MODE,
        ),
        (
            "blocksharedcalmass64",
            direct
            .PACKED_QKBALANCED_AUTOKEY_BLOCKSHAREDCALMASS64_FULLTOPK_SCORE_MODE,
        ),
        (
            "blocksharedcalmass128",
            direct
            .PACKED_QKBALANCED_AUTOKEY_BLOCKSHAREDCALMASS128_FULLTOPK_SCORE_MODE,
        ),
    ):
        VARIANTS[
            f"qksieve_keymse_requestlocal_{_name}_fulltopk_k{_budget}"
        ] = _score_mode
    for _rank, _score_mode in (
        (
            8,
            direct.PACKED_QKBALANCED_AUTOKEY_VALUESKETCH8I4_FULLTOPK_SCORE_MODE,
        ),
        (
            16,
            direct.PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_FULLTOPK_SCORE_MODE,
        ),
        (
            32,
            direct.PACKED_QKBALANCED_AUTOKEY_VALUESKETCH32I4_FULLTOPK_SCORE_MODE,
        ),
    ):
        VARIANTS[
            f"qksieve_keymse_requestlocal_valuesketch{_rank}i4_"
            f"fulltopk_k{_budget}"
        ] = _score_mode
        if _rank == 16:
            VARIANTS[
                "qksieve_keymse_requestlocal_valuesketch16i4_wometric_"
                f"fulltopk_k{_budget}"
            ] = (
                direct
                .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_FULLTOPK_SCORE_MODE
            )
            for _mass_permille, _mass_mode in (
                (
                    900,
                    direct
                    .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR900_FULLTOPK_SCORE_MODE,
                ),
                (
                    950,
                    direct
                    .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR950_FULLTOPK_SCORE_MODE,
                ),
                (
                    975,
                    direct
                    .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR975_FULLTOPK_SCORE_MODE,
                ),
            ):
                VARIANTS[
                    "qksieve_keymse_requestlocal_valuesketch16i4_wometric_"
                    f"massfloor{_mass_permille}_fulltopk_k{_budget}"
                ] = _mass_mode
            VARIANTS[
                "qksieve_keymse_requestlocal_valuesketch16i4_wometric_"
                f"massfloor950_affineres_fulltopk_k{_budget}"
            ] = (
                direct
                .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR950_AFFINERES_FULLTOPK_SCORE_MODE
            )
            VARIANTS[
                "qksieve_keymse_requestlocal_valuesketch16i4_wometric_"
                f"massfloor950_condres8_fulltopk_k{_budget}"
            ] = (
                direct
                .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_MASSFLOOR950_CONDRES8_FULLTOPK_SCORE_MODE
            )
            for _risk_bits, _risk_mode in (
                (
                    8,
                    direct
                    .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK8_FULLTOPK_SCORE_MODE,
                ),
                (
                    4,
                    direct
                    .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_FULLTOPK_SCORE_MODE,
                ),
            ):
                VARIANTS[
                    "qksieve_keymse_requestlocal_valuesketch16i4_wometric_"
                    f"residualrisk{_risk_bits}_fulltopk_k{_budget}"
                ] = _risk_mode
            for _rss_safety, _rss_mode in (
                (
                    1,
                    direct
                    .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_RSSREL5M_S1_FULLTOPK_SCORE_MODE,
                ),
                (
                    2,
                    direct
                    .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_RSSREL5M_S2_FULLTOPK_SCORE_MODE,
                ),
            ):
                VARIANTS[
                    "qksieve_keymse_requestlocal_valuesketch16i4_wometric_"
                    "residualrisk4_rssrel5m_"
                    f"safety{_rss_safety}_fulltopk_k{_budget}"
                ] = _rss_mode
            VARIANTS[
                "qksieve_keymse_requestlocal_valuesketch16i4_wometric_"
                "residualrisk4_globalfloorrss25e4_"
                f"safety1_fulltopk_k{_budget}"
            ] = (
                direct
                .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALFLOORRSS25E4_S1_FULLTOPK_SCORE_MODE
            )
            VARIANTS[
                "qksieve_keymse_requestlocal_valuesketch16i4_wometric_"
                "residualrisk4_prefixrss25e4_"
                f"safety1_fulltopk_k{_budget}"
            ] = (
                direct
                .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_PREFIXRSS25E4_S1_FULLTOPK_SCORE_MODE
            )
            VARIANTS[
                "qksieve_keymse_requestlocal_valuesketch16i4_wometric_"
                f"residualrisk4_globalalloc_fulltopk_k{_budget}"
            ] = (
                direct
                .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALALLOC_FULLTOPK_SCORE_MODE
            )
            VARIANTS[
                "qksieve_keymse_requestlocal_valuesketch16i4_wometric_"
                f"residualrisk4_globalcal256_globalalloc_fulltopk_k{_budget}"
            ] = (
                direct
                .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALCAL256_FULLTOPK_SCORE_MODE
            )
            for _anchor_mass, _anchor_mode in (
                (
                    30,
                    direct
                    .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALANCHOR30_FULLTOPK_SCORE_MODE,
                ),
                (
                    50,
                    direct
                    .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALANCHOR50_FULLTOPK_SCORE_MODE,
                ),
                (
                    70,
                    direct
                    .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALANCHOR70_FULLTOPK_SCORE_MODE,
                ),
            ):
                VARIANTS[
                    "qksieve_keymse_requestlocal_valuesketch16i4_wometric_"
                    f"residualrisk4_globalanchor{_anchor_mass}_"
                    f"fulltopk_k{_budget}"
                ] = _anchor_mode
        if _rank == 32:
            VARIANTS[
                "qksieve_keymse_requestlocal_valuesketch32i4_wometric_"
                f"fulltopk_k{_budget}"
            ] = (
                direct
                .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH32I4_WOMETRIC_FULLTOPK_SCORE_MODE
            )
    VARIANTS[
        "qksieve_keymse_requestlocal_valuesketch16i4_"
        f"layer0r128_fulltopk_k{_budget}"
    ] = (
        direct
        .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_LAYER0R128_FULLTOPK_SCORE_MODE
    )
    VARIANTS[
        "qksieve_keymse_requestlocal_valuesketch16i4_"
        f"condres8global_fulltopk_k{_budget}"
    ] = (
        direct
        .PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_CONDRES8_GLOBAL_FULLTOPK_SCORE_MODE
    )
    VARIANTS[
        f"qksieve_keymse_requestlocal_sampled_k{_budget}_c32"
    ] = direct.PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE
    VARIANTS[
        f"qksieve_keymse_requestlocal_valuesketch8i4_sampled_k{_budget}"
    ] = (
        direct
        .PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH8_SCORE_MODE
    )
    VARIANTS[
        f"qksieve_keymse_requestlocal_valuesketch16i4_sampled_k{_budget}"
    ] = (
        direct
        .PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH16_SCORE_MODE
    )
    VARIANTS[
        f"qksieve_keymse_requestlocal_valuesketch32i4_sampled_k{_budget}"
    ] = (
        direct
        .PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH32_SCORE_MODE
    )
    VARIANTS[
        f"qksieve_keymse_requestlocal_valuesketch8to32i4_sampled_k{_budget}"
    ] = (
        direct
        .PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_VALUESKETCH8TO32_SCORE_MODE
    )
    VARIANTS[
        f"qksieve_keymse_requestlocal_proxymass_k{_budget}_c32"
    ] = QKSIEVE_KEYMSE_PROXYMASS_SCORE_MODE
    VARIANTS[
        f"qksieve_keymse_requestlocal_meantail_k{_budget}_c32"
    ] = QKSIEVE_KEYMSE_MEANTAIL_SCORE_MODE
    VARIANTS[
        f"qksieve_keymse_requestlocal_tilttail16_k{_budget}_c32"
    ] = QKSIEVE_KEYMSE_TILTTAIL16_SCORE_MODE
    VARIANTS[
        f"qksieve_keymse_requestlocal_condtail32_k{_budget}_c32"
    ] = QKSIEVE_KEYMSE_CONDTAIL32_SCORE_MODE
    VARIANTS[
        f"qksieve_keymse_requestlocal_condtail16_k{_budget}_c32"
    ] = QKSIEVE_KEYMSE_CONDTAIL16_SCORE_MODE
    VARIANTS[
        f"qksieve_keymse_requestlocal_condtail8_k{_budget}_c32"
    ] = QKSIEVE_KEYMSE_CONDTAIL8_SCORE_MODE
    VARIANTS[
        f"qksieve_keymse_requestlocal_fixedalloc_fulltopk_k{_budget}"
    ] = direct.PACKED_QKBALANCED_FIXED41111100_FULLTOPK_SCORE_MODE
    VARIANTS[
        f"qksieve_keymse_frozenbasis_realloc_fulltopk_k{_budget}"
    ] = direct.PACKED_QKBALANCED_AUTOKEY_REALLOC_FULLTOPK_SCORE_MODE
    VARIANTS[f"qksieve_keymse_highbit_fulltopk_k{_budget}"] = (
        direct.PACKED_QKBALANCED_AUTOKEY_FULLTOPK_SCORE_MODE
    )

KEYMSE_BUDGETS = (
    1280,
    2560,
    3840,
    5120,
    7680,
    10240,
    12800,
    15728,
    20972,
    26214,
    31456,
    41933,
    52416,
    62900,
)
for _budget in KEYMSE_BUDGETS:
    VARIANTS[f"qksieve_deploy_keymse_k{_budget}"] = (
        direct.PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE
    )
    VARIANTS[f"qksieve_keymse_fulltopk_k{_budget}"] = (
        direct.PACKED_QKBALANCED_AUTOKEY_FULLTOPK_SCORE_MODE
    )
KEYMSE_SAMPLE_COUNTS = (512, 1024, 2048, 4096, 8192)
for _sample_count in KEYMSE_SAMPLE_COUNTS:
    VARIANTS[f"qksieve_deploy_keymse_k1280_s{_sample_count}"] = (
        direct.PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE
    )
for _budget in (7680, 10240, 12800, 15728, 20972, 26214, 31456):
    for _sample_count in KEYMSE_SAMPLE_COUNTS:
        VARIANTS[
            f"qksieve_deploy_keymse_k{_budget}_s{_sample_count}"
        ] = direct.PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE
for _budget, _sample_count in (
    (2560, 4096),
    (5120, 2048),
    (7680, 2048),
    (10240, 1024),
):
    VARIANTS[f"qksieve_deploy_keymse_k{_budget}_s{_sample_count}"] = (
        direct.PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE
    )
for _budget in KEYMSE_BUDGETS:
    VARIANTS[f"qksieve_deploy_keymse_k{_budget}_c64"] = (
        direct.PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_SCORE_MODE
    )

for _budget in (1120, 1280, 2560):
    VARIANTS[
        f"qksieve_qmse_oas_requestlocal_blockcondres8_r8_m8_k{_budget}"
    ] = (
        direct.PACKED_QKSIEVE_QMSE_OAS_BLOCKCONDRES8_R8_M8_FULLTOPK_SCORE_MODE
    )


def variant_integer(
    variant: str,
    marker: str,
    default: int,
) -> int:
    match = re.search(rf"{re.escape(marker)}(\d+)(?:_|$)", variant)
    if match is None:
        return default
    return int(match.group(1))


def variant_attention_token_budget(
    variant: str,
    history_tokens: int,
) -> int:
    """Resolve the frozen length-only budget while preserving diagnostic caps."""
    maximum = variant_integer(variant, "_k", 1280)
    if variant.startswith(
        "qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c"
    ) or variant.startswith(
        "qksieve_qmse_oas_requestlocal_blockcondres8_r8_m8_"
    ):
        return direct_countcap_target_count(
            history_tokens,
            fraction=0.06,
            min_tokens=256,
            max_tokens=maximum,
        )
    return min(history_tokens, maximum)


def binarypc_overfetch_factor(variant: str) -> float:
    match = re.search(r"exactrerank(\d+(?:\.\d+)?)x(?:_|$)", variant)
    if match is None:
        return 4.0
    return float(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument(
        "--binarypc_projection_path",
        type=Path,
        help="Official per-layer BinaryPC projection checkpoint.",
    )
    parser.add_argument(
        "--highbit_template",
        type=Path,
        help=(
            "Frozen QK-balanced template with a diagnostic high-bit "
            "allocation, used only by qksieve_keymse_highbit_* variants."
        ),
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--history_tokens", required=True, type=int)
    parser.add_argument(
        "--stream_reference_history_tokens",
        type=int,
        default=0,
        help=(
            "Build one deterministic stream up to this reference history "
            "position, keep its following eval tokens fixed, and use only "
            "the requested history_tokens-long suffix as model history. "
            "Zero preserves the original behavior."
        ),
    )
    parser.add_argument("--eval_tokens", type=int, default=16)
    parser.add_argument(
        "--collect_qk_product_spectrum",
        action="store_true",
        help="Measure request-local Cq/Ck product spectra after shared prefill.",
    )
    parser.add_argument(
        "--capture_layer_hidden_drift",
        action="store_true",
        help=(
            "Capture each decoder layer's final-token hidden state during "
            "evaluation and compare sparse variants with Full attention."
        ),
    )
    parser.add_argument("--topic", default="mixed_b")
    parser.add_argument(
        "--text_file",
        type=Path,
        help=(
            "Use one local UTF-8 text file as the deterministic evaluation "
            "stream instead of downloading the topic corpus."
        ),
    )
    parser.add_argument(
        "--synthetic_rope_seed",
        type=int,
        default=-1,
        help=(
            "Build the controlled two-hop RoPE retrieval stream instead of "
            "loading a natural-text topic. A nonnegative value is the case seed."
        ),
    )
    parser.add_argument(
        "--synthetic_rope_source_root",
        type=Path,
        default=Path(
            "/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary"
        ),
    )
    parser.add_argument("--prefill_chunk_tokens", type=int, default=4096)
    parser.add_argument("--protect_recent_tokens", type=int, default=0)
    parser.add_argument("--dataset_cache_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="balanced")
    parser.add_argument("--max_memory_per_gpu_gib", type=float, default=22.0)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--original_max_position_embeddings",
        type=int,
        default=0,
    )
    parser.add_argument("--global_max_position", type=int, default=0)
    parser.add_argument(
        "--variants",
        default="qksieve,coldskip50,coldskip60",
    )
    parser.add_argument(
        "--full_only",
        action="store_true",
        help="Evaluate only the dense Full-attention reference path.",
    )
    parser.add_argument(
        "--allow_context_extrapolation",
        action="store_true",
    )
    parser.add_argument(
        "--repeat_topic_stream_if_short",
        action="store_true",
        help=(
            "Deterministically reshuffle and cycle the same topic documents "
            "when one corpus pass is shorter than the requested window."
        ),
    )
    return parser.parse_args()


def write_progress(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@contextmanager
def temporary_environment(name: str, value: str | None):
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def decoder_layers(model: torch.nn.Module) -> Any:
    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        nested = getattr(decoder, "model", None)
        layers = getattr(nested, "layers", None)
    if layers is None:
        raise TypeError("model does not expose decoder layers")
    return layers


@contextmanager
def capture_layer_hidden_states(
    model: torch.nn.Module,
    enabled: bool,
):
    """Capture post-layer final-token states without changing model outputs."""

    captured: dict[int, list[torch.Tensor]] = defaultdict(list)
    if not enabled:
        yield captured
        return
    handles = []
    for layer_index, layer in enumerate(decoder_layers(model)):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            layer_index: int = layer_index,
        ) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_index].append(
                hidden[:, -1, :].detach().float().cpu()
            )

        handles.append(layer.register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def summarize_layer_hidden_drift(
    reference: dict[int, list[torch.Tensor]],
    candidate: dict[int, list[torch.Tensor]],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    if set(reference) != set(candidate):
        raise RuntimeError("Full and sparse runs captured different layers")
    for layer in sorted(reference):
        full_steps = reference[layer]
        sparse_steps = candidate[layer]
        if len(full_steps) != len(sparse_steps):
            raise RuntimeError(
                f"layer {layer} captured a different number of decode steps"
            )
        relative_l2 = []
        cosine = []
        absolute_l2 = []
        for full, sparse in zip(full_steps, sparse_steps):
            full = full.reshape(-1)
            sparse = sparse.reshape(-1)
            delta = torch.linalg.vector_norm(sparse - full)
            full_norm = torch.linalg.vector_norm(full).clamp_min(1.0e-12)
            sparse_norm = torch.linalg.vector_norm(sparse).clamp_min(1.0e-12)
            relative_l2.append(float(delta / full_norm))
            absolute_l2.append(float(delta))
            cosine.append(float(torch.dot(full, sparse) / (full_norm * sparse_norm)))
        rows.append(
            {
                "layer": layer,
                "decode_steps": len(full_steps),
                "relative_l2_mean": sum(relative_l2) / len(relative_l2),
                "relative_l2_max": max(relative_l2),
                "absolute_l2_mean": sum(absolute_l2) / len(absolute_l2),
                "cosine_mean": sum(cosine) / len(cosine),
                "cosine_min": min(cosine),
            }
        )
    return rows


def crop_cache(cache: Any, target_length: int) -> None:
    crop = getattr(cache, "crop", None)
    if not callable(crop):
        raise TypeError("shared prefill cache does not support crop()")
    crop(target_length)
    actual = int(cache.get_seq_length())
    if actual != target_length:
        raise RuntimeError(
            f"cache rollback failed: expected {target_length}, got {actual}"
        )


def peak_memory() -> list[int]:
    return [
        int(torch.cuda.max_memory_allocated(device))
        for device in range(torch.cuda.device_count())
    ]


def token_ids_sha256(token_ids: list[int]) -> str:
    payload = ",".join(str(int(token_id)) for token_id in token_ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def build_synthetic_rope_stream(
    tokenizer: Any,
    desired_history_tokens: int,
    seed: int,
    source_root: Path,
) -> tuple[list[int], list[int], dict[str, Any]]:
    source_dir = source_root / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    import run_local_global_rope_probe_8b as rope_probe

    body_length = max(512, int(desired_history_tokens) - 128)
    case: dict[str, Any] | None = None
    prompt_ids: list[int] = []
    target_prompt_count = int(desired_history_tokens) + 1
    for _ in range(3):
        case = rope_probe.seeded_case(tokenizer, body_length, seed)
        prompt_ids = case["prompt"].reshape(-1).tolist()
        difference = target_prompt_count - len(prompt_ids)
        if difference == 0:
            break
        body_length = max(512, body_length + difference)
    if case is None or len(prompt_ids) != target_prompt_count:
        raise RuntimeError(
            "could not construct the requested synthetic RoPE history: "
            f"wanted prompt={target_prompt_count}, got={len(prompt_ids)}"
        )
    gold = str(case["codes"][-1])
    gold_ids = tokenizer(
        f" {gold}",
        add_special_tokens=False,
    )["input_ids"]
    if len(gold_ids) != 1:
        raise RuntimeError(
            f"synthetic RoPE answer is not one token: {gold!r} -> {gold_ids}"
        )
    return (
        prompt_ids[:-1],
        [prompt_ids[-1], int(gold_ids[0])],
        {
            "seed": int(seed),
            "gold": gold,
            "codes": list(case["codes"]),
            "evidence_spans": [
                [int(start), int(end)]
                for start, end in case["evidence_spans"]
            ],
            "query_token_id": int(prompt_ids[-1]),
            "gold_token_id": int(gold_ids[0]),
            "gold_target_index": 1,
        },
    )


def annotate_synthetic_gold_metrics(
    summary: dict[str, Any],
    token_rows: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
    logits: list[torch.Tensor] | None = None,
) -> None:
    if metadata is None:
        return
    target_index = int(metadata["gold_target_index"])
    row = next(
        (
            value
            for value in token_rows
            if int(value["target_index"]) == target_index
        ),
        None,
    )
    if row is None:
        raise RuntimeError("synthetic gold target row was not evaluated")
    gold_nll = float(row["nll"])
    summary["synthetic_gold_nll"] = gold_nll
    summary["synthetic_gold_ppl"] = math.exp(gold_nll)
    summary["synthetic_gold_probability"] = math.exp(-gold_nll)
    top1: int | None = None
    if logits is not None:
        top1 = int(logits[target_index].reshape(-1).argmax().item())
    elif row.get("sparse_top1_id") is not None:
        top1 = int(row["sparse_top1_id"])
    if top1 is not None:
        summary["synthetic_gold_correct"] = int(
            top1 == int(metadata["gold_token_id"])
        )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.history_tokens <= 1 or args.eval_tokens <= 1:
        raise ValueError("history_tokens and eval_tokens must exceed one")
    stream_reference_history_tokens = int(
        args.stream_reference_history_tokens or args.history_tokens
    )
    if stream_reference_history_tokens < args.history_tokens:
        raise ValueError(
            "stream_reference_history_tokens must be at least history_tokens"
        )
    if (
        args.synthetic_rope_seed >= 0
        and stream_reference_history_tokens != args.history_tokens
    ):
        raise ValueError(
            "fixed-target history slicing is not implemented for synthetic "
            "RoPE streams"
        )
    variants = [] if args.full_only else [
        value.strip() for value in args.variants.split(",") if value.strip()
    ]
    unknown = sorted(set(variants) - set(VARIANTS))
    if (not variants and not args.full_only) or unknown:
        raise ValueError(f"unsupported variants: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    install_llama_head_top_fraction_patch()
    tokenizer, model, input_device = load_model(args)
    args.model = model
    preload_extensions = not args.full_only and (
        os.environ.get("QKSIEVE_PRELOAD_EXTENSIONS", "").strip().lower()
        in {"1", "true", "yes"}
    )
    extension_preload_timings = (
        preload_qksieve_runtime_extensions() if preload_extensions else {}
    )
    preload_qmse_rate_tables = preload_extensions and (
        os.environ.get("QKSIEVE_PRELOAD_QMSE_RATE_TABLES", "1")
        .strip()
        .lower()
        not in {"0", "false", "no"}
    )
    qmse_rate_table_preload = (
        preload_qksieve_qmse_rate_tables(model)
        if preload_qmse_rate_tables
        else {}
    )
    value_metric_precompute = (
        precompute_qksieve_value_metric_grams(model)
        if preload_extensions
        else {}
    )
    args.binarypc_projections = None
    if any(value.startswith("binarypc_exactrerank") for value in variants):
        if args.binarypc_projection_path is None:
            raise ValueError(
                "BinaryPC exact-rerank variants require "
                "--binarypc_projection_path"
            )
        projections = torch.load(
            args.binarypc_projection_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(projections, dict):
            raise TypeError("BinaryPC projection checkpoint must be a dict")
        args.binarypc_projections = {
            int(layer): projection
            for layer, projection in projections.items()
            if isinstance(projection, torch.Tensor)
        }
    args.direct_score_mode = VARIANTS["qksieve"]
    args.packed_qmse_template_in = args.template
    args.packed_qmse_template_out = None
    args.direct_fraction = 0.06
    args.exact_fraction = 0.02
    args.direct_min_tokens = 256
    args.direct_max_tokens = 1280
    args.projection_dim = 128
    args.sample_count = 256
    args.candidate_overfetch = 1.0
    args.qk_metric_query_shrinkage = 0.75
    args.cache_mode = "preallocated"
    args.preallocated_cache_min_tokens = 1
    args.collect_logit_stability = True
    native_context = int(
        getattr(model.config, "max_position_embeddings", 0) or 0
    )
    required_tokens = args.history_tokens + args.eval_tokens
    stream_required_tokens = (
        stream_reference_history_tokens + args.eval_tokens
    )
    extrapolated = (
        native_context > 0 and required_tokens > native_context
    )
    if extrapolated and not args.allow_context_extrapolation:
        raise ValueError(
            f"{required_tokens} required tokens exceed native context "
            f"{native_context}; "
            "pass --allow_context_extrapolation for a stress test"
        )

    synthetic_metadata: dict[str, Any] | None = None
    if args.synthetic_rope_seed >= 0:
        history, target_ids, synthetic_metadata = build_synthetic_rope_stream(
            tokenizer,
            args.history_tokens,
            args.synthetic_rope_seed,
            args.synthetic_rope_source_root,
        )
    elif args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8")
        stream = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
        if not stream:
            raise RuntimeError(f"text stream is empty: {args.text_file}")
        if len(stream) < stream_required_tokens:
            if not args.repeat_topic_stream_if_short:
                raise RuntimeError(
                    f"text stream has {len(stream)} tokens, need "
                    f"{stream_required_tokens}; pass "
                    "--repeat_topic_stream_if_short to cycle it"
                )
            repeats = math.ceil(stream_required_tokens / len(stream))
            stream = (stream * repeats)[:stream_required_tokens]
        else:
            stream = stream[:stream_required_tokens]
        history_start = (
            stream_reference_history_tokens - args.history_tokens
        )
        history = stream[
            history_start:stream_reference_history_tokens
        ]
        target_ids = stream[
            stream_reference_history_tokens:
            stream_reference_history_tokens + args.eval_tokens
        ]
    else:
        stream = direct.encode_evaluation_stream(
            tokenizer,
            args.topic,
            stream_required_tokens,
            args.dataset_cache_dir,
            args.seed,
            repeat_documents=args.repeat_topic_stream_if_short,
        )
        history_start = (
            stream_reference_history_tokens - args.history_tokens
        )
        history = stream[
            history_start:stream_reference_history_tokens
        ]
        target_ids = stream[
            stream_reference_history_tokens:
            stream_reference_history_tokens + args.eval_tokens
        ]
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)

    print(
        f"[prefill] history={args.history_tokens} "
        f"chunk={args.prefill_chunk_tokens}",
        flush=True,
    )
    prefill_started = time.perf_counter()
    with prefill_query_tail_mode(8) as prefill_queries:
        cache, previous_logits, dense_seconds = direct.dense_prompt(
            model,
            tokenizer,
            history,
            input_device,
            args.prefill_chunk_tokens,
            "preallocated",
            1,
            len(target_ids),
        )
    shared_prefill_seconds = time.perf_counter() - prefill_started
    if int(cache.get_seq_length()) != args.history_tokens:
        raise RuntimeError("shared prefill produced the wrong cache length")
    shared_prefill = (
        cache,
        previous_logits,
        dense_seconds,
        prefill_queries,
    )
    build_resident_value_sketch = not args.full_only and (
        os.environ.get("QKSIEVE_BUILD_RESIDENT_VALUE_SKETCH", "")
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )
    resident_value_sketch_precompute: dict[str, float | int] = {}
    build_resident_key_factors = not args.full_only and (
        os.environ.get("QKSIEVE_BUILD_RESIDENT_KEY_FACTORS", "")
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )
    resident_key_factor_precompute: dict[str, float | int | str] = {}
    if build_resident_value_sketch or build_resident_key_factors:
        score_modes = {VARIANTS[variant] for variant in variants}
        if len(score_modes) != 1 or EXACT_FP16_QK_ORACLE in score_modes:
            raise ValueError(
                "resident prebuild requires one non-oracle mode"
            )
        args.direct_score_mode = next(iter(score_modes))
        with direct.sparse_context(args, "direct_countcap"):
            if build_resident_value_sketch:
                resident_workers = int(
                    os.environ.get("QKSIEVE_RESIDENT_VALUE_WORKERS", "12")
                )
                resident_value_sketch_precompute = (
                    prebuild_resident_value_sketch_cache(
                        cache,
                        model,
                        max_workers=resident_workers,
                    )
                )
            if build_resident_key_factors:
                resident_key_workers = int(
                    os.environ.get("QKSIEVE_RESIDENT_KEY_WORKERS", "36")
                )
                resident_key_factor_precompute = (
                    prebuild_resident_qksieve_key_factors(
                        cache,
                        max_workers=resident_key_workers,
                    )
                )
    qk_product_spectrum = (
        analyze_cache_qk_product_spectrum(
            cache,
            prefill_queries,
            sample_stride=32,
            query_shrinkages=(0.75,),
        )
        if args.collect_qk_product_spectrum
        else None
    )

    payload: dict[str, Any] = {
        "schema": "qksieve_coldskip_longcontext_quality_v1",
        "model_name_or_path": str(args.model_name_or_path),
        "template": str(args.template),
        "highbit_template": (
            str(args.highbit_template)
            if args.highbit_template is not None
            else None
        ),
        "binarypc_projection_path": (
            str(args.binarypc_projection_path)
            if args.binarypc_projection_path is not None
            else None
        ),
        "history_tokens": args.history_tokens,
        "stream_reference_history_tokens": (
            stream_reference_history_tokens
        ),
        "fixed_target_history_start": (
            stream_reference_history_tokens - args.history_tokens
        ),
        "target_token_ids_sha256": token_ids_sha256(target_ids),
        "recent_256_token_ids_sha256": token_ids_sha256(history[-256:]),
        "eval_tokens": len(target_ids),
        "topic": args.topic,
        "text_file": (
            str(args.text_file.resolve())
            if args.text_file is not None
            else None
        ),
        "topic_stream_repeat_if_short": bool(
            args.repeat_topic_stream_if_short
        ),
        "synthetic_rope": synthetic_metadata,
        "seed": args.seed,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "device_map": args.device_map,
        "gpu_count": torch.cuda.device_count(),
        "requested_variants": variants,
        "native_context_tokens": native_context,
        "context_extrapolation": extrapolated,
        "shared_prefill_seconds": shared_prefill_seconds,
        "extension_preload_seconds": extension_preload_timings,
        "qmse_rate_table_preload": qmse_rate_table_preload,
        "value_metric_precompute": value_metric_precompute,
        "resident_value_sketch_precompute": (
            resident_value_sketch_precompute
        ),
        "resident_key_factor_precompute": resident_key_factor_precompute,
        "prefill_query_layers": len(prefill_queries),
        "prefill_query_tokens": min(
            (
                int(value.shape[-2])
                for value in prefill_queries.values()
            ),
            default=0,
        ),
        "rows": [],
        "token_rows": {},
        "qk_product_spectrum": qk_product_spectrum,
        "capture_layer_hidden_drift": bool(
            args.capture_layer_hidden_drift
        ),
        "peak_gpu_allocated_bytes_after_prefill": peak_memory(),
    }
    progress_path = args.output_dir / "summary.json"
    write_progress(progress_path, payload)

    with capture_layer_hidden_states(
        model,
        args.capture_layer_hidden_drift,
    ) as full_hidden_states:
        full_summary, full_token_rows, full_logits = direct.evaluate_method(
            args,
            tokenizer,
            history,
            target_ids,
            "full_attention",
            input_device,
            capture_logits=True,
            shared_prefill=shared_prefill,
        )
    crop_cache(cache, args.history_tokens)
    annotate_synthetic_gold_metrics(
        full_summary,
        full_token_rows,
        synthetic_metadata,
        full_logits,
    )
    full_summary["variant"] = "full_attention"
    payload["rows"].append(full_summary)
    payload["token_rows"]["full_attention"] = full_token_rows
    write_progress(progress_path, payload)
    print(json.dumps(full_summary, sort_keys=True), flush=True)

    if full_logits is None:
        raise RuntimeError("Full logits were not captured")
    for variant in variants:
        oracle_exact_qk = VARIANTS[variant] == EXACT_FP16_QK_ORACLE
        if not oracle_exact_qk:
            args.direct_score_mode = VARIANTS[variant]
        binarypc_variant = variant.startswith("binarypc_exactrerank")
        binarypc_overfetch = binarypc_overfetch_factor(variant)
        request_local = variant.startswith(
            (
                "qksieve_keymse_requestlocal_",
                "qksieve_qmse_requestlocal_",
                "qksieve_qmse_oas_requestlocal_",
            )
        )
        request_local_fixedalloc = variant.startswith(
            "qksieve_keymse_requestlocal_fixedalloc_"
        )
        request_local_fp16x2 = variant.startswith(
            "qksieve_requestlocal_fp16x2_"
        )
        proxy_mass_variant = "_proxymass_" in variant
        progressive_value_sketch = "_valuesketch8to32" in variant
        post2x_prerope_rerank = (
            "_post2xprererank_" in variant
            or "_post2xboundary75prererank_" in variant
            or "_post2xdualmass_" in variant
        )
        dual_mass_prerope_rerank = "_post2xdualmass_" in variant
        boundary75_prerope_rerank = (
            "_post2xboundary75prererank_" in variant
        )
        full_prerope_index = "_fullprerope_" in variant
        prerope32 = "_prerope32" in variant
        frozen_basis_realloc = variant.startswith(
            "qksieve_keymse_frozenbasis_realloc_"
        )
        highbit_frozen = variant.startswith(
            "qksieve_keymse_highbit_fulltopk_"
        )
        if binarypc_variant or request_local or request_local_fp16x2:
            args.packed_qmse_template_in = None
        elif highbit_frozen:
            if args.highbit_template is None:
                raise ValueError(
                    f"{variant} requires --highbit_template"
                )
            args.packed_qmse_template_in = args.highbit_template
        else:
            args.packed_qmse_template_in = args.template
        args.direct_max_tokens = variant_attention_token_budget(
            variant,
            args.history_tokens,
        )
        args.candidate_overfetch = (
            binarypc_overfetch if binarypc_variant else 1.0
        )
        args.exact_tokens = args.direct_max_tokens
        args.direct_fraction = min(
            1.0,
            args.direct_max_tokens / args.history_tokens,
        )
        target_tail_samples = variant_integer(variant, "_c", 0)
        if oracle_exact_qk:
            args.sample_count = 0
        elif target_tail_samples > 0:
            target_fraction = args.direct_max_tokens / args.history_tokens
            args.sample_count = direct.tail_resolution_sample_count(
                target_tail_samples,
                target_fraction,
            )
        else:
            args.sample_count = variant_integer(variant, "_s", 256)
        disable_value_sketch = "_novalue_" in variant
        with temporary_environment(
            "QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH",
            "1" if disable_value_sketch else None,
        ):
            with capture_layer_hidden_states(
                model,
                args.capture_layer_hidden_drift,
            ) as sparse_hidden_states:
                summary, token_rows, _ = direct.evaluate_method(
                    args,
                    tokenizer,
                    history,
                    target_ids,
                    "exact_top_k" if oracle_exact_qk else "direct_countcap",
                    input_device,
                    reference_logits=full_logits,
                    shared_prefill=shared_prefill,
                )
        if args.capture_layer_hidden_drift:
            summary["layer_hidden_drift"] = summarize_layer_hidden_drift(
                full_hidden_states,
                sparse_hidden_states,
            )
        crop_cache(cache, args.history_tokens)
        annotate_synthetic_gold_metrics(
            summary,
            token_rows,
            synthetic_metadata,
        )
        summary["variant"] = variant
        summary["value_sketch_disabled_ablation"] = disable_value_sketch
        summary["max_exact_tokens_per_head"] = args.direct_max_tokens
        summary["max_exact_fraction"] = (
            args.direct_max_tokens / args.history_tokens
        )
        summary["requested_attention_fraction"] = args.direct_fraction
        summary["requested_quantile_sample_count_per_head"] = (
            args.sample_count
        )
        summary["target_tail_sample_count"] = target_tail_samples
        summary["selector"] = (
            "exact_fp16_qk_topk"
            if oracle_exact_qk
            else (
                f"binarypc64_overfetch{binarypc_overfetch:g}x_"
                "exact_qk_rerank"
            )
            if binarypc_variant
            else "qksieve_proxy_mass_progressive_value_rank8_to_rank32"
            if progressive_value_sketch
            else "pre_rope_lowfreq32_quantized_candidates_exact_attention"
            if prerope32
            else "qksieve_post2x_exact_pre_post_mass_mixture_exact_attention"
            if dual_mass_prerope_rerank
            else "qksieve_post2x_pool_exact_prerope_rerank_exact_attention"
            if post2x_prerope_rerank and not boundary75_prerope_rerank
            else "qksieve_post2x_boundary75_exact_prerope_rerank_exact_attention"
            if boundary75_prerope_rerank
            else "qksieve_full_prerope_lowbit_proposal_exact_postrope_attention"
            if full_prerope_index
            else "qkbalanced_requestlocal_fixedallocation_lowbit_proxy"
            if request_local_fixedalloc
            else "qkbalanced_requestlocal_sampled_proxy_mass_corrected"
            if proxy_mass_variant
            else "qkbalanced_requestlocal_fp16_first_two_band_proxy"
            if request_local_fp16x2
            else "qkbalanced_requestlocal_lowbit_proxy"
            if request_local
            else "qkbalanced_frozenbasis_reallocated_lowbit_proxy"
            if frozen_basis_realloc
            else "qkbalanced_frozen_highbit_proxy"
            if highbit_frozen
            else "qkbalanced_frozen_lowbit_proxy"
        )
        summary["template_policy"] = (
            "none_exact_qk"
            if oracle_exact_qk
            else "official_binarypc64_projection"
            if binarypc_variant
            else "no_pca_template_pre_rope_frequency_coordinates"
            if prerope32
            else "request_local_fixed_allocation"
            if request_local_fixedalloc
            else "request_local_fp16_first_two_bands"
            if request_local_fp16x2
            else "request_local"
            if request_local
            else "frozen_basis_requestlocal_allocation"
            if frozen_basis_realloc
            else "frozen_highbit"
            if highbit_frozen
            else "frozen_default"
        )
        summary["delta_nll_vs_full"] = (
            float(summary["nll"]) - float(full_summary["nll"])
        )
        summary["quality_retention"] = math.exp(
            float(full_summary["nll"]) - float(summary["nll"])
        )
        if synthetic_metadata is not None:
            summary["synthetic_gold_quality_retention"] = math.exp(
                float(full_summary["synthetic_gold_nll"])
                - float(summary["synthetic_gold_nll"])
            )
        payload["rows"].append(summary)
        payload["token_rows"][variant] = token_rows
        payload["peak_gpu_allocated_bytes_after_methods"] = peak_memory()
        write_progress(progress_path, payload)
        print(json.dumps(summary, sort_keys=True), flush=True)
        torch.cuda.empty_cache()

    coldskip_rows = [
        row
        for row in payload["rows"]
        if row["variant"].startswith("coldskip")
    ]
    if coldskip_rows:
        qksieve = next(
            row for row in payload["rows"] if row["variant"] == "qksieve"
        )
        for row in coldskip_rows:
            row["quality_retention_vs_qksieve"] = math.exp(
                float(qksieve["nll"]) - float(row["nll"])
            )
    write_progress(progress_path, payload)
    (args.output_dir / "ALL_COMPLETE").touch()
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
