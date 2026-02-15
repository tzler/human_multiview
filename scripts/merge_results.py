#!/usr/bin/env python3
"""Merge per-model confidence CSVs into a single all_models_confidence.csv.

Use after running parallel evaluation jobs:
    python scripts/merge_results.py [--results_dir results/]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from human_multiview.config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = args.results_dir
    model_files = sorted([
        f for f in os.listdir(results_dir)
        if f.endswith("_confidence.csv")
        and f != "all_models_confidence.csv"
        and "_backup" not in f
    ])

    if not model_files:
        # Fall back to backup files
        model_files = sorted([
            f for f in os.listdir(results_dir)
            if f.endswith("_confidence_backup.csv")
        ])

    if not model_files:
        print("No per-model confidence CSVs found.")
        return

    print(f"Found {len(model_files)} model files: {model_files}")

    merged = None
    for fname in model_files:
        df = pd.read_csv(os.path.join(results_dir, fname))
        model_name = fname.replace("_confidence_backup.csv", "").replace("_confidence.csv", "")
        print(f"  {model_name}: {len(df)} trials, columns: {[c for c in df.columns if '_correct' in c]}")

        if merged is None:
            merged = df
        else:
            model_cols = [c for c in df.columns if c not in merged.columns]
            if model_cols:
                merged = merged.merge(df[["trial_name"] + model_cols], on="trial_name", how="outer")

    if merged is not None:
        out_path = os.path.join(results_dir, "all_models_confidence.csv")
        merged.to_csv(out_path, index=False)
        print(f"\nMerged: {out_path} ({len(merged)} trials, {len(merged.columns)} columns)")
        print(f"Models: {set(c.split('_')[0] for c in merged.columns if '_correct' in c)}")


if __name__ == "__main__":
    main()
