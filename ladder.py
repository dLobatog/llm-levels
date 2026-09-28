"""Shows progress: a level is unlocked when every test in its file passes.

Also reports style: the code should follow the Google Python Style Guide,
checked by ruff with the configuration in pyproject.toml.

Typical usage example:

    uv run python ladder.py            # the whole ladder
    uv run python ladder.py --current  # just the level you're on, in detail
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent

LEVELS = [
    (0, "Setup", None),
    (1, "Tokenizer", "tests/test_level01_tokenizer.py"),
    (2, "Data and the training loop", "tests/test_level02_training_loop.py"),
    (3, "Attention", "tests/test_level03_attention.py"),
    (4, "GPT-2", "tests/test_level04_gpt2.py"),
    (5, "Pretraining", "tests/test_level05_pretraining.py"),
    (6, "Inference and modern blocks", "tests/test_level06_inference.py"),
    (7, "Fine-tuning", "tests/test_level07_finetuning.py"),
    (8, "Preference optimization", "tests/test_level08_dpo.py"),
    (9, "RL with verifiable rewards", "tests/test_level09_grpo.py"),
    (10, "Stretch", None),
]


def _run(*args: str) -> str:
    """Runs a Python module from the repo root and returns its stdout."""
    command = [sys.executable, "-m", *args]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout


def run_tests(path: str) -> tuple[int, int]:
    """Runs one test file.

    Args:
        path: Test file, relative to the repo root.

    Returns:
        The number of tests that passed and the total number of tests.
    """
    out = _run("pytest", "-q", "--no-header", "-p", "no:cacheprovider", path)
    passed, failed = _count(out)
    return passed, passed + failed


def _count(out: str) -> tuple[int, int]:
    """Returns (passed, failed or errored) from pytest's summary line."""
    pattern = r"(\d+) (passed|failed|error|errors)"
    counts = {kind: int(count) for count, kind in re.findall(pattern, out)}
    failed = sum(counts.get(kind, 0) for kind in ("failed", "error", "errors"))
    return counts.get("passed", 0), failed


def run_current() -> None:
    """Shows pytest's output for the first level that doesn't pass yet.

    Earlier levels are rerun too, so a regression there takes focus. Stops at
    the first failure, which keeps the output short enough to read.
    """
    for number, name, tests in LEVELS:
        if tests is None or not (ROOT / tests).exists():
            continue
        out = _run(
            "pytest",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "--tb=short",
            "--color=yes",
            "-x",
            tests,
        )
        passed, failed = _count(out)
        if failed or not passed:
            print(f"▶ level {number} · {name}\n")
            print(out)
            return
        print(f"✓ level {number} · {name}  ({passed} tests)")
    print("\nevery level with tests passes: time for the next brief")


def count_style_issues() -> int:
    """Returns the count of lint findings plus files to reformat."""
    lint = _run("ruff", "check", "--quiet", "--output-format", "concise")
    unformatted = _run("ruff", "format", "--check", "--quiet")
    return len(lint.splitlines()) + len(unformatted.splitlines())


def main() -> None:
    """Prints one line per level, then the style status."""
    current_found = False
    for number, name, tests in LEVELS:
        label = f"{number:>2}  {name}"
        if number == 0:
            print(f"✓ {label}")
            continue
        if tests is None or not (ROOT / tests).exists():
            print(
                f"· {label:<34} brief arrives when the previous level is done"
            )
            continue
        passed, total = run_tests(tests)
        if total and passed == total:
            print(f"✓ {label:<34} {passed}/{total} tests")
        elif not current_found:
            current_found = True
            brief = next(ROOT.glob(f"levels/{number:02d}-*.md"), None)
            where = brief.relative_to(ROOT) if brief else "?"
            print(f"▶ {label:<34} {passed}/{total} tests   brief: {where}")
        else:
            print(f"· {label:<34} {passed}/{total} tests")

    issues = count_style_issues()
    if issues:
        print(f"\nstyle: {issues} issues (uv run ruff check; ruff format)")
    else:
        print("\nstyle: ✓ Google Python Style Guide (ruff)")


if __name__ == "__main__":
    if sys.argv[1:] == ["--current"]:
        run_current()
    else:
        main()
