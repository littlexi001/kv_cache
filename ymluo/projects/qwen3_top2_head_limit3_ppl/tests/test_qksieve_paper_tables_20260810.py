from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "papers" / "countcap_iclr2027" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "make_qksieve_quality_tables.py"
SPEC = importlib.util.spec_from_file_location("qksieve_paper_tables", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tables_render_actual_frozen_summary_schema() -> None:
    longbench = {
        "methods": {
            "full_kv": {"macro_score": 0.46},
            "qksieve_frozen": {
                "macro_score": 0.4554,
                "quality_retention": 0.99,
            },
        },
        "bootstrap": {"quality_retention_95ci": [0.985, 0.995]},
    }
    ruler = {
        "per_length": {
            str(length): {
                "full_macro": 0.9,
                "qksieve_macro": 0.891,
                "quality_retention": 0.99,
                "bootstrap": {"quality_retention_95ci": [0.98, 1.0]},
            }
            for length in (4096, 8192, 16384, 32768, 65536, 131072)
        },
        "overall": {
            "full_macro": 0.9,
            "qksieve_macro": 0.891,
            "quality_retention": 0.99,
        },
        "bootstrap": {"quality_retention_95ci": [0.985, 0.995]},
    }
    multimodel = {
        "models": {
            model: {
                "full_macro": 0.5,
                "qksieve_macro": 0.495,
                "quality_retention": 0.99,
                "quality_retention_95ci": [0.98, 1.0],
            }
            for model in MODULE.quality_figure.MODEL_ORDER
        }
    }

    text = MODULE.render(
        longbench,
        ruler,
        multimodel,
        chinese=False,
        provenance="longbench=abc; ruler=def; multimodel=ghi",
    )

    assert "Llama-3.1-8B & 0.4600 & 0.4554 & 99.00\\%" in text
    assert "4K & 0.9000 & 0.8910 & 99.00\\%" in text
    assert "Overall & 0.9000 & 0.8910 & 99.00\\%" in text
    assert "Mistral-7B & 0.5000 & 0.4950 & 99.00\\%" in text
    assert (
        "Generated from frozen evidence: longbench=abc; ruler=def; "
        "multimodel=ghi"
    ) in text

    main_text = MODULE.render_main(
        longbench,
        ruler,
        multimodel,
        chinese=False,
        provenance="longbench=abc; ruler=def; multimodel=ghi",
    )
    assert "Full LB / Llama-3.1-8B & 0.4600 & 0.4554 & 99.00\\%" in main_text
    assert "RULER / Llama-3.1-8B & 0.9000 & 0.8910 & 99.00\\%" in main_text
    assert "LB screen / Mistral-7B & 0.5000 & 0.4950 & 99.00\\%" in main_text
    assert "\\label{tab:quality-main}" in main_text
    assert "\\resizebox{\\columnwidth}{!}" in main_text
