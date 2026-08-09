from __future__ import annotations

import unittest

from src.run_controlled_public_kv_benchmark_v1 import split_multiple_choice_query


class ChoiceContrastQueryTest(unittest.TestCase):
    def test_splits_longbench_v2_query(self) -> None:
        query = (
            "What is the correct answer to this question: Which item is supported?\n"
            "Choices:\n"
            "(A) First answer\n"
            "(B) Second answer with (parentheses)\n"
            "(C) Third answer\n"
            "(D) Fourth answer"
        )
        stem, choices = split_multiple_choice_query(query)
        self.assertEqual(stem, "What is the correct answer to this question: Which item is supported?")
        self.assertEqual(
            choices,
            ["First answer", "Second answer with (parentheses)", "Third answer", "Fourth answer"],
        )

    def test_non_multiple_choice_query_is_unchanged(self) -> None:
        query = "Summarize this document."
        stem, choices = split_multiple_choice_query(query)
        self.assertEqual(stem, query)
        self.assertEqual(choices, [])


if __name__ == "__main__":
    unittest.main()
