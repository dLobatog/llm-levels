"""Where am I? A level is unlocked when every test in its file passes.

Usage: uv run python ladder.py
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


def run_tests(path: str) -> tuple[int, int]:
    """Return (passed, total) for one test file."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", path],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|error|errors)", out)}
    passed = counts.get("passed", 0)
    total = passed + counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
    return passed, total


def main() -> None:
    current_found = False
    for num, name, tests in LEVELS:
        label = f"{num:>2}  {name}"
        if num == 0:
            print(f"✓ {label}")
            continue
        if tests is None or not (ROOT / tests).exists():
            print(f"· {label:<34} brief arrives when the previous level is done")
            continue
        passed, total = run_tests(tests)
        if total and passed == total:
            print(f"✓ {label:<34} {passed}/{total} tests")
        elif not current_found:
            current_found = True
            brief = next(ROOT.glob(f"levels/{num:02d}-*.md"), None)
            print(f"▶ {label:<34} {passed}/{total} tests   brief: {brief.relative_to(ROOT) if brief else '?'}")
        else:
            print(f"· {label:<34} {passed}/{total} tests")


if __name__ == "__main__":
    main()
