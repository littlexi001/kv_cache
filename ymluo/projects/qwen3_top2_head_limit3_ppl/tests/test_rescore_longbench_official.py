from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from rescore_longbench_official_20260723 import rescore_row, stored_prediction


def test_rescore_escaped_triviaqa_first_line():
    row = {
        "task": "triviaqa",
        "metric": "qa_f1",
        "prediction": "Dartmoor\\nPassage:\\nunrelated",
        "answers": '["Dartmoor"]',
        "score": "0.1",
    }

    assert stored_prediction(row) == "Dartmoor"
    assert rescore_row(row)["score"] == 1.0


def test_rescore_leaves_regular_tasks_unchanged():
    row = {
        "task": "narrativeqa",
        "metric": "qa_f1",
        "prediction": "answer\\ncontinuation",
        "answers": '["answer"]',
        "score": "0.25",
    }

    assert rescore_row(row) == row


def test_rescore_preserves_already_scored_classification_row():
    row = {
        "task": "trec",
        "metric": "classification",
        "prediction": "Other location\\nQuestion: continuation",
        "answers": '["Other location"]',
        "score": "1.0",
    }

    assert rescore_row(row) == row
