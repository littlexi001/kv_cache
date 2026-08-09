from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np


DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit minute-level and day-level temporal causality in LongMemEval."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples", type=int, default=30)
    return parser.parse_args()


def quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = json.loads(
        Path(args.input).read_text(encoding="utf-8")
    )
    total_history_sessions = 0
    future_history_sessions_minute = 0
    future_history_sessions_day = 0
    queries_with_future_history_minute = 0
    queries_with_future_history_day = 0
    positive_queries = 0
    abstention_queries = 0
    evidence_sessions = 0
    evidence_future_minute = 0
    evidence_future_day = 0
    queries_with_future_evidence_minute = 0
    queries_with_future_evidence_day = 0
    future_evidence_by_type_minute: collections.Counter[str] = collections.Counter()
    future_evidence_by_type_day: collections.Counter[str] = collections.Counter()
    lead_hours_minute: list[float] = []
    lead_days_calendar: list[float] = []
    examples = []

    for row in rows:
        question_date = dt.datetime.strptime(str(row["question_date"]), DATE_FORMAT)
        session_dates = [
            dt.datetime.strptime(str(value), DATE_FORMAT)
            for value in row["haystack_dates"]
        ]
        total_history_sessions += len(session_dates)
        future_history_minute = [value for value in session_dates if value > question_date]
        future_history_day = [
            value for value in session_dates if value.date() > question_date.date()
        ]
        future_history_sessions_minute += len(future_history_minute)
        future_history_sessions_day += len(future_history_day)
        queries_with_future_history_minute += int(bool(future_history_minute))
        queries_with_future_history_day += int(bool(future_history_day))

        is_abstention = str(row["question_id"]).endswith("_abs")
        positive_queries += int(not is_abstention)
        abstention_queries += int(is_abstention)
        if is_abstention:
            continue
        session_date_by_id = {
            str(session_id): value
            for session_id, value in zip(row["haystack_session_ids"], session_dates)
        }
        answer_dates = [
            session_date_by_id[str(session_id)]
            for session_id in row["answer_session_ids"]
        ]
        evidence_sessions += len(answer_dates)
        minute_violations = [value for value in answer_dates if value > question_date]
        day_violations = [
            value for value in answer_dates if value.date() > question_date.date()
        ]
        evidence_future_minute += len(minute_violations)
        evidence_future_day += len(day_violations)
        if minute_violations:
            queries_with_future_evidence_minute += 1
            future_evidence_by_type_minute[str(row["question_type"])] += 1
            lead_hours_minute.extend(
                (value - question_date).total_seconds() / 3600.0
                for value in minute_violations
            )
            if len(examples) < args.examples:
                examples.append(
                    {
                        "question_id": str(row["question_id"]),
                        "question_type": str(row["question_type"]),
                        "question_date": str(row["question_date"]),
                        "future_evidence_dates": [
                            value.strftime(DATE_FORMAT) for value in minute_violations
                        ],
                        "lead_hours": [
                            (value - question_date).total_seconds() / 3600.0
                            for value in minute_violations
                        ],
                    }
                )
        if day_violations:
            queries_with_future_evidence_day += 1
            future_evidence_by_type_day[str(row["question_type"])] += 1
            lead_days_calendar.extend(
                float((value.date() - question_date.date()).days)
                for value in day_violations
            )

    output = {
        "source": "LongMemEval-S full temporal-causality audit",
        "questions": len(rows),
        "positive_queries": positive_queries,
        "abstention_queries": abstention_queries,
        "history_sessions": total_history_sessions,
        "minute_level": {
            "queries_with_future_history": queries_with_future_history_minute,
            "future_history_sessions": future_history_sessions_minute,
            "queries_with_future_positive_evidence": queries_with_future_evidence_minute,
            "future_positive_evidence_sessions": evidence_future_minute,
            "future_evidence_query_fraction": (
                queries_with_future_evidence_minute / positive_queries
            ),
            "future_evidence_session_fraction": evidence_future_minute / evidence_sessions,
            "future_evidence_queries_by_type": dict(future_evidence_by_type_minute),
            "lead_hours": quantiles(lead_hours_minute),
        },
        "calendar_day_level": {
            "queries_with_future_history": queries_with_future_history_day,
            "future_history_sessions": future_history_sessions_day,
            "queries_with_future_positive_evidence": queries_with_future_evidence_day,
            "future_positive_evidence_sessions": evidence_future_day,
            "future_evidence_queries_by_type": dict(future_evidence_by_type_day),
            "lead_calendar_days": quantiles(lead_days_calendar),
        },
        "interpretation": (
            "Minute-level violations confined to the same calendar day are timestamp-order "
            "noise for day-level tasks, but must still be excluded from strict online-causal "
            "experiments. Later-calendar-day violations are genuine future leakage."
        ),
        "examples": examples,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
