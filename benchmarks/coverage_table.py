"""The coverage table of the scoring intervals, as quoted in docs/validation.md.

python benchmarks/coverage_table.py    (about ten minutes: 1000 repeats per cell)
"""

from vbar.eval.coverage import (
    iqm_coverage,
    mcnemar_false_positives,
    paired_coverage,
    rate_coverage,
)

SEEDS = (5, 10, 20, 50, 100)


def main():
    columns = (
        "Success 0.9",
        "Keep-out 0.05",
        "IQM",
        "Paired difference",
        "McNemar false positives",
    )
    print("| Runs per cell | " + " | ".join(columns) + " |")
    print("|---|" + "---|" * len(columns))
    for n in SEEDS:
        cells = [
            rate_coverage(0.9, n, 1000, rng=1),
            rate_coverage(0.05, n, 1000, rng=2),
            iqm_coverage(n, 1000, 1000, rng=3),
            paired_coverage(n, 1000, 1000, rng=4),
        ]
        text = [f"{c.rate:.3f}{'' if c.holds else ' (short)'}" for c in cells]
        false = mcnemar_false_positives(0.8, n, 1000, rng=5)
        print(f"| {n} | " + " | ".join(text) + f" | {false.estimate:.3f} |")
    narrow = iqm_coverage(100, 1000, 1000, rng=3, widen=0.7)
    print(f"\nIQM interval narrowed to 0.7 of its width, 100 runs: {narrow.rate:.3f}")


if __name__ == "__main__":
    main()
