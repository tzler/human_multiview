#!/usr/bin/env python3
"""Run model evaluation on the MOCHI benchmark.

This script evaluates multi-view models and a DINOv2 baseline on the MOCHI
3D perception benchmark, producing CSVs that the analysis notebooks consume.

Usage:
    # Run all models on all trials
    python scripts/run_evaluation.py --gpu_id 2

    # Run specific models
    python scripts/run_evaluation.py --models vggt dinov2 --gpu_id 2

    # Quick test run
    python scripts/run_evaluation.py --models vggt --n_trials 10 --gpu_id 2

Outputs (in results/):
    all_models_confidence.csv   — per-trial confidence-based accuracy for all models
    vggt_layer_analysis.csv     — per-trial layer-wise accuracy + solution layers
    vggt_confidence_margins.csv — per-trial confidence margins
    dinov2_similarity.csv       — per-trial DINOv2 cosine similarity accuracy
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add parent directory so we can import human_multiview
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from human_multiview.config import RESULTS_DIR
from human_multiview.data import load_mochi
from human_multiview.models import get_model
from human_multiview.models.base import clear_gpu_memory
from human_multiview.evaluate import (
    evaluate_trial_confidence,
    evaluate_trial_dinov2,
    evaluate_trial_layers,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on MOCHI benchmark")
    parser.add_argument(
        "--models", nargs="+",
        default=["vggt", "dust3r", "mast3r", "pi3"],
        help="Models to evaluate (default: all multi-view models)",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--n_trials", type=int, default=None, help="Number of trials (default: all)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--skip_majaj", action="store_true", default=True,
                        help="Skip majaj dataset trials (default: True)")
    return parser.parse_args()


def run_confidence_evaluation(model_name, device, dataset, trial_indices):
    """Evaluate a single model using confidence-based oddity prediction."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'='*60}")

    ModelClass = get_model(model_name)
    model = ModelClass()
    model.load(device)

    rows = []
    for trial_idx in tqdm(trial_indices, desc=model_name):
        trial = dataset[int(trial_idx)]
        try:
            results = evaluate_trial_confidence(model, trial["images"])
            row = {
                "trial_name": trial["trial"],
                "dataset": trial["dataset"],
                "condition": trial.get("condition", trial["dataset"]),
                "oddity_index": trial["oddity_index"],
                "n_images": len(trial["images"]),
            }
            for key, res in results.items():
                row[f"{model_name}_{key}_correct"] = res["predicted_oddity"] == trial["oddity_index"]
                row[f"{model_name}_{key}_margin"] = res["confidence_margin"]
            rows.append(row)
        except Exception as e:
            print(f"  Error on {trial['trial']}: {e}")
            continue

    model.unload()
    return pd.DataFrame(rows)


def run_dinov2_evaluation(device, dataset, trial_indices):
    """Evaluate DINOv2 baseline using cosine similarity."""
    print(f"\n{'='*60}")
    print(f"Evaluating DINOv2 (baseline)")
    print(f"{'='*60}")

    ModelClass = get_model("dinov2")
    model = ModelClass()
    model.load(device)

    rows = []
    for trial_idx in tqdm(trial_indices, desc="dinov2"):
        trial = dataset[int(trial_idx)]
        try:
            results = evaluate_trial_dinov2(model, trial["images"], trial["oddity_index"])
            row = {
                "trial_name": trial["trial"],
                "dataset": trial["dataset"],
                "condition": trial.get("condition", trial["dataset"]),
                "oddity_index": trial["oddity_index"],
            }
            for metric, correct in results.items():
                row[f"dinov2_{metric}"] = correct
            rows.append(row)
        except Exception as e:
            print(f"  Error on {trial['trial']}: {e}")
            continue

    model.unload()
    return pd.DataFrame(rows)


def run_layer_analysis(device, dataset, trial_indices):
    """Run VGGT layer-wise analysis for solution layer computation."""
    print(f"\n{'='*60}")
    print(f"Running VGGT layer-wise analysis")
    print(f"{'='*60}")

    ModelClass = get_model("vggt")
    model = ModelClass()
    model.load(device)

    rows = []
    for trial_idx in tqdm(trial_indices, desc="vggt layers"):
        trial = dataset[int(trial_idx)]
        try:
            results = evaluate_trial_layers(
                model, trial["images"], trial["oddity_index"]
            )
            row = {
                "trial_name": trial["trial"],
                "dataset": trial["dataset"],
                "condition": trial.get("condition", trial["dataset"]),
                "oddity_index": trial["oddity_index"],
                "human_accuracy": trial.get("human_avg", None),
                "human_rt": trial.get("RT_avg", None),
            }
            # Per-layer correctness
            for metric, layer_results in results["per_layer"].items():
                for layer_idx, correct in layer_results.items():
                    row[f"vggt_L{layer_idx}_{metric}"] = correct
            # Solution layers
            for metric, sol_layer in results["solution_layers"].items():
                row[f"earliest_reliable_layer_{metric}"] = sol_layer
            rows.append(row)
        except Exception as e:
            print(f"  Error on {trial['trial']}: {e}")
            continue

    model.unload()
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if "cuda" in device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = "cuda"

    output_dir = args.output_dir or str(RESULTS_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Models: {args.models}")

    # Load dataset
    print("\nLoading MOCHI dataset...")
    dataset = load_mochi()
    print(f"Loaded {len(dataset)} trials")

    # Select trial indices
    all_indices = list(range(len(dataset)))
    if args.skip_majaj:
        all_indices = [i for i in all_indices if "majaj" not in dataset[i]["dataset"]]
        print(f"After excluding majaj: {len(all_indices)} trials")
    if args.n_trials is not None:
        np.random.seed(42)
        all_indices = np.random.choice(all_indices, args.n_trials, replace=False).tolist()
        print(f"Sampling {args.n_trials} trials")

    # --- 1. Confidence-based evaluation for all multi-view models ---
    confidence_dfs = []
    for model_name in args.models:
        if model_name == "dinov2":
            continue  # handled separately
        df = run_confidence_evaluation(model_name, device, dataset, all_indices)
        confidence_dfs.append(df)
        clear_gpu_memory()

    if confidence_dfs:
        # Save per-model confidence files (safe for parallel runs)
        for df in confidence_dfs:
            model_cols = [c for c in df.columns if "_correct" in c or "_margin" in c]
            if model_cols:
                model_name = model_cols[0].split("_")[0]
                per_model_path = os.path.join(output_dir, f"{model_name}_confidence.csv")
                df.to_csv(per_model_path, index=False)
                print(f"Saved: {per_model_path} ({len(df)} trials)")

        # Merge all model results on trial_name
        merged = confidence_dfs[0]
        for df in confidence_dfs[1:]:
            model_cols = [c for c in df.columns if c not in merged.columns]
            merged = merged.merge(df[["trial_name"] + model_cols], on="trial_name", how="outer")

        # Also merge any existing per-model CSVs from parallel runs
        for fname in os.listdir(output_dir):
            if fname.endswith("_confidence.csv") and fname != "all_models_confidence.csv":
                model_prefix = fname.replace("_confidence.csv", "")
                if not any(c.startswith(f"{model_prefix}_") for c in merged.columns if "_correct" in c):
                    other = pd.read_csv(os.path.join(output_dir, fname))
                    other_cols = [c for c in other.columns if c not in merged.columns]
                    if other_cols:
                        merged = merged.merge(other[["trial_name"] + other_cols], on="trial_name", how="outer")
                        print(f"Merged results from {fname}")

        # Add human data
        trial_meta = {}
        for i in all_indices:
            t = dataset[int(i)]
            trial_meta[t["trial"]] = {
                "human_accuracy": t.get("human_avg", None),
                "human_rt": t.get("RT_avg", None),
            }
        merged["human_accuracy"] = merged["trial_name"].map(lambda n: trial_meta.get(n, {}).get("human_accuracy"))
        merged["human_rt"] = merged["trial_name"].map(lambda n: trial_meta.get(n, {}).get("human_rt"))

        path = os.path.join(output_dir, "all_models_confidence.csv")
        merged.to_csv(path, index=False)
        print(f"\nSaved: {path} ({len(merged)} trials)")

    # --- 2. DINOv2 baseline ---
    if "dinov2" in args.models or "vggt" in args.models:
        df_dino = run_dinov2_evaluation(device, dataset, all_indices)
        path = os.path.join(output_dir, "dinov2_similarity.csv")
        df_dino.to_csv(path, index=False)
        print(f"Saved: {path}")
        clear_gpu_memory()

    # --- 3. VGGT layer-wise analysis ---
    if "vggt" in args.models:
        df_layers = run_layer_analysis(device, dataset, all_indices)
        path = os.path.join(output_dir, "vggt_layer_analysis.csv")
        df_layers.to_csv(path, index=False)
        print(f"Saved: {path}")
        clear_gpu_memory()

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
