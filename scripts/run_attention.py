#!/usr/bin/env python3
"""Extract attention data from VGGT for visualization.

This script extracts cross-image attention maps for selected trials and saves
them to disk so the attention visualization notebook can work with them.

Usage:
    # Extract attention for default example trials
    python scripts/run_attention.py --gpu_id 2

    # Extract attention for specific trials (by index)
    python scripts/run_attention.py --trial_indices 60 18 811 --gpu_id 2

    # Extract attention for specific layers
    python scripts/run_attention.py --layers 0 15 23 --gpu_id 2

Outputs (in results/attention/):
    trial_{idx}_attention.npz  — attention maps + patch_info per trial
    trial_{idx}_images.npz     — trial images as numpy arrays
    trial_{idx}_meta.npz       — trial metadata (name, dataset, oddity_index)
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

# Add parent directory so we can import human_multiview
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from human_multiview.config import RESULTS_DIR
from human_multiview.data import load_mochi, get_trial_images
from human_multiview.models import get_model
from human_multiview.models.base import clear_gpu_memory
from human_multiview.attention import AttentionExtractor, extract_attention


def parse_args():
    parser = argparse.ArgumentParser(description="Extract VGGT attention maps")
    parser.add_argument(
        "--trial_indices", nargs="+", type=int, default=None,
        help="Trial indices to extract (default: one per condition)",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=None,
        help="Layer indices to extract (default: all 24 layers)",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    return parser.parse_args()


def select_default_trials(dataset):
    """Select one trial per condition for default extraction."""
    seen = {}
    for i in range(len(dataset)):
        condition = dataset[i]["dataset"]
        if condition not in seen and "majaj" not in condition:
            seen[condition] = i
    return sorted(seen.values())


def save_trial_data(output_dir, trial_idx, trial, attention_maps_pairs, patch_info, layer_indices):
    """Save attention data and images for one trial."""
    img_A, img_Ap, img_B, trial_info = get_trial_images(trial)

    # Save images
    np.savez_compressed(
        os.path.join(output_dir, f"trial_{trial_idx}_images.npz"),
        img_A=np.array(img_A),
        img_Ap=np.array(img_Ap),
        img_B=np.array(img_B),
    )

    # Save attention maps for each pair (A→A' and A→B)
    attn_data = {}
    for pair_name, attn_maps in attention_maps_pairs.items():
        for layer_idx, attn_tensor in attn_maps.items():
            attn_data[f"{pair_name}_layer{layer_idx}"] = attn_tensor.numpy()

    attn_data["patch_start_idx"] = patch_info["patch_start_idx"]
    attn_data["num_patches_per_image"] = patch_info["num_patches_per_image"]
    attn_data["patch_h"] = patch_info["patch_h"]
    attn_data["patch_w"] = patch_info["patch_w"]
    attn_data["total_tokens"] = patch_info["total_tokens"]
    attn_data["layers"] = np.array(layer_indices)

    np.savez_compressed(
        os.path.join(output_dir, f"trial_{trial_idx}_attention.npz"),
        **attn_data,
    )

    # Save metadata
    np.savez(
        os.path.join(output_dir, f"trial_{trial_idx}_meta.npz"),
        trial_name=trial_info["trial_name"],
        dataset=trial_info["dataset"],
        condition=trial_info["condition"],
        oddity_index=trial_info["oddity_index"],
    )


def main():
    args = parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if "cuda" in device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = "cuda"

    output_dir = args.output_dir or str(RESULTS_DIR / "attention")
    os.makedirs(output_dir, exist_ok=True)

    layer_indices = args.layers if args.layers else list(range(24))

    # Load dataset
    print("Loading MOCHI dataset...")
    dataset = load_mochi()
    print(f"Loaded {len(dataset)} trials")

    # Select trials
    if args.trial_indices is not None:
        trial_indices = args.trial_indices
    else:
        trial_indices = select_default_trials(dataset)
    print(f"Extracting attention for {len(trial_indices)} trials")
    print(f"Layers: {layer_indices}")

    # Load VGGT
    print("\nLoading VGGT...")
    ModelClass = get_model("vggt")
    vggt_model = ModelClass()
    vggt_model.load(device)

    extractor = AttentionExtractor(vggt_model.model)

    for trial_idx in trial_indices:
        trial = dataset[int(trial_idx)]
        trial_name = trial["trial"]
        print(f"\n  Trial {trial_idx}: {trial_name} ({trial['dataset']})")

        img_A, img_Ap, img_B, _ = get_trial_images(trial)

        # Extract attention for A→A' pair
        attn_maps_Ap, patch_info = extract_attention(
            vggt_model, img_A, img_Ap, extractor, tuple(layer_indices)
        )

        # Extract attention for A→B pair
        attn_maps_B, _ = extract_attention(
            vggt_model, img_A, img_B, extractor, tuple(layer_indices)
        )

        save_trial_data(
            output_dir, trial_idx, trial,
            {"A_Ap": attn_maps_Ap, "A_B": attn_maps_B},
            patch_info, layer_indices,
        )
        print(f"    Saved to {output_dir}/trial_{trial_idx}_*.npz")

        del attn_maps_Ap, attn_maps_B
        clear_gpu_memory()

    vggt_model.unload()

    print(f"\n{'='*60}")
    print(f"Attention extraction complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
