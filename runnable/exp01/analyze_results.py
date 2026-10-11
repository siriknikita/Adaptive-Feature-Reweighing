
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import statistics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_json")
    args = p.parse_args()

    data = json.loads(Path(args.results_json).read_text())
    experts = data["experts"]
    rows = data["results"]

    print("Experts:", ", ".join(experts))
    print()

    for r in rows:
        ws = ", ".join(f"{n}={w:.3f}" for n, w in zip(experts, r["mean_weights"]))
        print(
            f'{r["corruption"]:8s} | {r["method"]:18s} | '
            f'k={r["k_shot"]:2d} | acc={r["accuracy"]:.4f} | {ws}'
        )

    base = next((r for r in rows if r["corruption"] == "glasses" and r["method"] == "global"), None)
    best = None
    for r in rows:
        if r["corruption"] == "glasses" and r["method"] == "adaptive_few_shot":
            if best is None or r["accuracy"] > best["accuracy"]:
                best = r

    print()
    if base and best:
        delta = best["accuracy"] - base["accuracy"]
        print(f"Best few-shot improvement over global on glasses: {delta:+.4f}")
        print(f"Best K: {best['k_shot']}")
    else:
        print("Could not compute the comparison.")

if __name__ == "__main__":
    main()
