#!/usr/bin/env python3
import json
import os
import statistics
from collections import Counter


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = os.path.join(repo_root, "maps", "town2.json")

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Cannot find town2.json at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    quads = data.get("quads", [])
    if not quads:
        print("No quads found in town2.json")
        return

    curvatures = [float(q.get("curvature", 0.0)) for q in quads]
    n = len(curvatures)

    curvatures_sorted = sorted(curvatures)
    mean_val = statistics.fmean(curvatures)
    median_val = statistics.median(curvatures)
    min_val = curvatures_sorted[0]
    max_val = curvatures_sorted[-1]
    std_val = statistics.pstdev(curvatures) if n > 1 else 0.0

    percentiles = {
        "p05": percentile(curvatures_sorted, 5),
        "p25": percentile(curvatures_sorted, 25),
        "p50": percentile(curvatures_sorted, 50),
        "p75": percentile(curvatures_sorted, 75),
        "p90": percentile(curvatures_sorted, 90),
        "p95": percentile(curvatures_sorted, 95),
        "p99": percentile(curvatures_sorted, 99),
    }

    zero_count = sum(1 for v in curvatures if abs(v) < 1e-9)
    positive_count = sum(1 for v in curvatures if v > 0)
    negative_count = sum(1 for v in curvatures if v < 0)

    bucket_counter = Counter(round(v, 3) for v in curvatures)
    most_common = bucket_counter.most_common(10)

    print(f"Total quads: {n}")
    print(f"Mean curvature: {mean_val:.6f}")
    print(f"Median curvature: {median_val:.6f}")
    print(f"Min curvature: {min_val:.6f}")
    print(f"Max curvature: {max_val:.6f}")
    print(f"Std curvature: {std_val:.6f}")
    print(f"Zero curvature count: {zero_count}")
    print(f"Positive curvature count: {positive_count}")
    print(f"Negative curvature count: {negative_count}")
    print("Percentiles:")
    for name, value in percentiles.items():
        print(f"  {name}: {value:.6f}")
    print("Top 10 most common rounded curvatures (rounded to 1e-3):")
    for value, count in most_common:
        print(f"  {value:.3f}: {count}")


if __name__ == "__main__":
    main()

