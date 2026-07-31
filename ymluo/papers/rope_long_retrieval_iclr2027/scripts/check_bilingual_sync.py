"""Check structural synchronization between the English and Chinese editions.

The prose is intentionally different, but labels, references, citations,
displayed mathematics, numeric literals, environments, and TODO counts must
stay aligned. Run this before compiling after editing either edition.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "01_introduction.tex",
    "02_problem_setup.tex",
    "03_mechanism.tex",
    "04_method.tex",
    "05_experiments.tex",
    "06_related_work.tex",
    "07_conclusion.tex",
    "appendix.tex",
]
COMMANDS = ("label", "ref", "cref", "Cref", "cite", "citep", "citet")
MATH_ENVS = ("equation", "equation*", "align", "align*", "gather", "gather*")


def normalize_math(value: str) -> str:
    return re.sub(r"\s+", "", value)


def command_args(text: str, command: str) -> Counter[str]:
    pattern = re.compile(rf"\\{command}(?:\[[^\]]*\])?\{{([^}}]+)\}}")
    return Counter(pattern.findall(text))


def math_blocks(text: str) -> Counter[str]:
    blocks: list[str] = []
    for env in MATH_ENVS:
        pattern = re.compile(
            rf"\\begin\{{{re.escape(env)}\}}(.*?)\\end\{{{re.escape(env)}\}}",
            re.DOTALL,
        )
        blocks.extend(normalize_math(match) for match in pattern.findall(text))
    return Counter(blocks)


def numeric_literals(text: str) -> Counter[str]:
    pattern = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:,\d{{3}})*(?:\.\d+)?(?:\\%|%)?")
    # Bare 0/1 can arise from ordinary-language choices (e.g. "first" versus
    # "第 1"); all measured values and larger layer/length indices remain strict.
    return Counter(token for token in pattern.findall(text) if token not in {"0", "1"})


def environments(text: str, kind: str) -> Counter[str]:
    return Counter(re.findall(rf"\\{kind}\{{([^}}]+)\}}", text))


def compare(name: str, left, right, errors: list[str], filename: str) -> None:
    if left != right:
        errors.append(f"{filename}: {name} differs\n  English: {left}\n  Chinese: {right}")


def main() -> None:
    errors: list[str] = []
    for filename in FILES:
        english = (ROOT / "sections" / filename).read_text(encoding="utf-8")
        chinese = (ROOT / "sections_zh" / filename).read_text(encoding="utf-8")

        for command in COMMANDS:
            compare(
                f"\\{command} arguments",
                command_args(english, command),
                command_args(chinese, command),
                errors,
                filename,
            )
        compare("displayed math", math_blocks(english), math_blocks(chinese), errors, filename)
        compare("numeric literals", numeric_literals(english), numeric_literals(chinese), errors, filename)
        compare("begin environments", environments(english, "begin"), environments(chinese, "begin"), errors, filename)
        compare("end environments", environments(english, "end"), environments(chinese, "end"), errors, filename)
        compare("TODO count", english.count("\\todoexp{"), chinese.count("\\todoexp{"), errors, filename)

    if errors:
        raise SystemExit("Bilingual synchronization check failed:\n\n" + "\n\n".join(errors))
    print(f"Bilingual synchronization check passed for {len(FILES)} section pairs.")


if __name__ == "__main__":
    main()
