#!/usr/bin/env python3
"""
Parse Minix output and convert to JSON format for comparison.
"""

import re
import json
import sys

def parse_minix_output(output_file):
    """Parse Minix benchmark output."""
    with open(output_file, 'r') as f:
        content = f.read()

    results = []

    # Pattern: [N/136] NAME ... STATUS (iters, time_ms, ...)
    pattern = r'\[(\d+)/\d+\]\s+(\S+)\s+\.\.\.\s+([✓~MN])\s+\((\d+)\s+iters?,\s+([\d.]+)ms'

    for match in re.finditer(pattern, content):
        idx = int(match.group(1))
        name = match.group(2)
        status_symbol = match.group(3)
        iters = int(match.group(4))
        time_ms = float(match.group(5))

        status_map = {
            '✓': 'Optimal',
            '~': 'AlmostOptimal',
            'M': 'MaxIters',
            'N': 'NumericalError',
        }
        status = status_map.get(status_symbol, 'Unknown')

        results.append({
            "name": name,
            "status": status,
            "iterations": iters,
            "solve_time_ms": time_ms,
            "n": 0,
            "m": 0,
            "obj_val": 0.0,
            "mu": 0.0,
        })

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Minix output file")
    parser.add_argument("--output", required=True, help="JSON output file")
    args = parser.parse_args()

    results = parse_minix_output(args.input)

    # Compute summary
    optimal = sum(1 for r in results if r["status"] == "Optimal")
    almost = sum(1 for r in results if r["status"] == "AlmostOptimal")

    output = {
        "solver_name": "Minix",
        "results": results,
        "summary": {
            "total": len(results),
            "optimal": optimal,
            "almost_optimal": almost,
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Parsed {len(results)} results from {args.input}")
    print(f"Saved to {args.output}")
    print(f"Pass rate: {optimal} + {almost} = {optimal + almost}/{len(results)} ({100.0*(optimal+almost)/len(results):.1f}%)")


if __name__ == "__main__":
    main()
