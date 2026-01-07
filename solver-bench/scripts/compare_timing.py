#!/usr/bin/env python3
"""
Compare wall-clock time between Minix and Clarabel on commonly solved problems.

Usage:
    python compare_timing.py --minix /tmp/minix_1e9.json --clarabel /tmp/clarabel.json
"""

import argparse
import json
import numpy as np


def load_results(path):
    """Load benchmark results from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def analyze_common_problems(minix_data, clarabel_data):
    """Analyze problems that both solvers successfully solved."""

    # Build maps by problem name
    minix_map = {r["name"]: r for r in minix_data["results"]}
    clarabel_map = {r["name"]: r for r in clarabel_data["results"]}

    # Find problems solved by both (Optimal or AlmostOptimal)
    minix_solved = {name for name, r in minix_map.items()
                    if r["status"] in ["Optimal", "AlmostOptimal"]}
    clarabel_solved = {name for name, r in clarabel_map.items()
                       if r["status"] in ["Optimal", "AlmostOptimal"]}

    common_solved = sorted(minix_solved & clarabel_solved)

    print(f"Problems solved by both: {len(common_solved)}")
    print(f"Problems only Minix solved: {len(minix_solved - clarabel_solved)}")
    print(f"Problems only Clarabel solved: {len(clarabel_solved - minix_solved)}")
    print()

    if not common_solved:
        print("No common problems solved!")
        return

    # Collect timing data for common problems
    minix_times = []
    clarabel_times = []
    speedup_ratios = []

    for name in common_solved:
        mt = max(minix_map[name]["solve_time_ms"], 0.001)  # Avoid zero
        ct = max(clarabel_map[name]["solve_time_ms"], 0.001)
        minix_times.append(mt)
        clarabel_times.append(ct)
        speedup_ratios.append(ct / mt)  # > 1 means Minix is faster

    # Compute statistics
    minix_geom = np.exp(np.mean(np.log([t + 1.0 for t in minix_times]))) - 1.0
    clarabel_geom = np.exp(np.mean(np.log([t + 1.0 for t in clarabel_times]))) - 1.0
    overall_speedup = clarabel_geom / minix_geom

    print("=" * 80)
    print("TIMING COMPARISON (on commonly solved problems)")
    print("=" * 80)
    print()
    print(f"Number of problems: {len(common_solved)}")
    print()
    print(f"Minix geometric mean time:    {minix_geom:8.2f} ms")
    print(f"Clarabel geometric mean time: {clarabel_geom:8.2f} ms")
    print()
    print(f"Overall speedup: {overall_speedup:.2f}x")
    if overall_speedup > 1:
        print(f"  → Minix is {overall_speedup:.2f}x FASTER on average")
    else:
        print(f"  → Clarabel is {1/overall_speedup:.2f}x FASTER on average")
    print()

    # Show distribution
    faster_minix = sum(1 for r in speedup_ratios if r > 1.0)
    faster_clarabel = sum(1 for r in speedup_ratios if r < 1.0)

    print(f"Minix faster on: {faster_minix}/{len(common_solved)} problems ({100*faster_minix/len(common_solved):.1f}%)")
    print(f"Clarabel faster on: {faster_clarabel}/{len(common_solved)} problems ({100*faster_clarabel/len(common_solved):.1f}%)")
    print()

    # Show top 10 fastest for each
    sorted_by_speedup = sorted(zip(common_solved, speedup_ratios), key=lambda x: x[1], reverse=True)

    print("Top 10 problems where Minix is fastest (speedup > 1.0):")
    print(f"{'Problem':<15} {'Minix (ms)':>12} {'Clarabel (ms)':>15} {'Speedup':>10}")
    print("-" * 60)
    for name, speedup in sorted_by_speedup[:10]:
        if speedup > 1.0:
            print(f"{name:<15} {minix_map[name]['solve_time_ms']:>12.2f} {clarabel_map[name]['solve_time_ms']:>15.2f} {speedup:>9.2f}x")
    print()

    print("Top 10 problems where Clarabel is fastest (speedup < 1.0):")
    print(f"{'Problem':<15} {'Minix (ms)':>12} {'Clarabel (ms)':>15} {'Speedup':>10}")
    print("-" * 60)
    for name, speedup in sorted_by_speedup[-10:]:
        if speedup < 1.0:
            print(f"{name:<15} {minix_map[name]['solve_time_ms']:>12.2f} {clarabel_map[name]['solve_time_ms']:>15.2f} {1/speedup:>9.2f}x")
    print()

    # Return statistics
    return {
        "num_common": len(common_solved),
        "minix_geom_mean_ms": minix_geom,
        "clarabel_geom_mean_ms": clarabel_geom,
        "overall_speedup": overall_speedup,
        "faster_minix": faster_minix,
        "faster_clarabel": faster_clarabel,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare timing between Minix and Clarabel")
    parser.add_argument("--minix", required=True, help="Path to Minix results JSON")
    parser.add_argument("--clarabel", required=True, help="Path to Clarabel results JSON")
    args = parser.parse_args()

    minix_data = load_results(args.minix)
    clarabel_data = load_results(args.clarabel)

    analyze_common_problems(minix_data, clarabel_data)


if __name__ == "__main__":
    main()
