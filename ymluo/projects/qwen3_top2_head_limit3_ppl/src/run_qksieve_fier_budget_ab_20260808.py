#!/usr/bin/env python
"""Expose matched QKSieve/FIER variants through the frozen quality driver."""

from __future__ import annotations

import run_direct_countcap_denseprompt_ppl_20260725 as direct
import run_qksieve_coldskip_longcontext_quality_20260730 as experiment


experiment.VARIANTS[
    "qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k512"
] = direct.PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SORTED_SAMPLED_SCORE_MODE
experiment.VARIANTS[
    "qksieve_qmse_requestlocal_fier_rtn1_g32_fulltopk_k1280"
] = (
    direct.FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE
)
experiment.VARIANTS[
    "qksieve_qmse_requestlocal_fier_rtn1_g32_fulltopk_k512"
] = (
    direct.FIER_RTN1_G32_PACKED_FULLTOPK_SCORE_MODE
)


if __name__ == "__main__":
    experiment.main()
