from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_controlled_public_kv_benchmark_v1 import score_prediction
from rouge import Rouge


def test_triviaqa_uses_only_first_generated_line():
    prediction = "Dartmoor\nPassage:\nAn unrelated continuation"
    answers = ["Dartmoor"]

    assert score_prediction(
        "qa_f1", prediction, answers, task="triviaqa"
    ) == 1.0
    assert score_prediction("qa_f1", prediction, answers) < 1.0


def test_samsum_uses_only_first_generated_line():
    prediction = "Alice will meet Bob.\nDialogue: unrelated text"
    answers = ["Alice will meet Bob."]

    assert score_prediction(
        "rouge_l", prediction, answers, task="samsum"
    ) == pytest.approx(1.0)
    assert score_prediction("rouge_l", prediction, answers) < 1.0


def test_regular_qa_keeps_multiline_answer():
    prediction = "first line\nsecond line"
    answers = ["second line"]

    assert score_prediction(
        "qa_f1", prediction, answers, task="narrativeqa"
    ) > 0.0


def test_rouge_l_matches_official_longbench_package():
    prediction = "The cats were running through a city park."
    answer = "A cat runs through the park."
    expected = Rouge().get_scores(
        [prediction],
        [answer],
        avg=True,
    )["rouge-l"]["f"]

    assert score_prediction(
        "rouge_l", prediction, [answer], task="gov_report"
    ) == expected
