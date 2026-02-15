# Human-level 3D shape perception emerges from multi-view learning

Code and analysis pipeline for [Bonnen, Malik, & Kanazawa (2025)](https://arxiv.org/).

We evaluate multi-view vision transformers (VGGT, DUST3R, MAST3R, Pi3) and a single-image baseline (DINOv2) on the [MOCHI](https://huggingface.co/datasets/tzler/MOCHI) 3D perception benchmark alongside human behavioral data.

## Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/tzler/human_multiview.git
cd human_multiview

# Create conda environment
conda env create -f environment.yml
conda activate human_multiview

# Install this package
pip install -e .
```

## Usage

The repo is a complete pipeline: **scripts generate data, notebooks visualize data**.

### 1. Run model evaluation

```bash
# Run all models on all MOCHI trials (~2000 trials)
python scripts/run_evaluation.py --gpu_id 0

# Run specific models
python scripts/run_evaluation.py --models vggt dinov2 --gpu_id 0

# Quick test (10 trials)
python scripts/run_evaluation.py --models vggt --n_trials 10 --gpu_id 0
```

This produces CSVs in `results/`:
- `all_models_confidence.csv` — per-trial confidence-based accuracy for all multi-view models
- `dinov2_similarity.csv` — per-trial DINOv2 cosine similarity accuracy
- `vggt_layer_analysis.csv` — per-trial layer-wise accuracy + solution layers

### 2. Extract attention data (optional, for Fig 4)

```bash
python scripts/run_attention.py --gpu_id 0
```

### 3. Generate figures

Open the notebooks to reproduce all manuscript figures:

| Notebook | Figures | GPU required |
|----------|---------|:---:|
| `notebooks/01_main_results.ipynb` | Fig 3, S1–S6 | No |
| `notebooks/02_layer_analysis.ipynb` | Fig 2 (bottom), S7 | Yes (for Fig 2) |
| `notebooks/03_attention_visualization.ipynb` | Fig 4, S8, S9 | Yes |

## Project structure

```
human_multiview/          Python package
├── config.py             Constants and model registry
├── data.py               MOCHI dataset loading
├── models/               Model wrappers (one per model)
│   ├── base.py           Abstract base class
│   ├── vggt.py           VGGT-1B
│   ├── dust3r.py         DUST3R
│   ├── mast3r.py         MAST3R
│   ├── pi3.py            Pi3
│   └── dinov2.py         DINOv2 baseline
├── evaluate.py           Oddity prediction, confidence margins, solution layers
├── attention.py          Cross-image attention extraction and visualization
└── plotting.py           Figure styling

scripts/                  Data generation
├── run_evaluation.py     Run all models → CSVs
└── run_attention.py      Extract attention maps → NPZs

notebooks/                Figure generation
├── 01_main_results.ipynb
├── 02_layer_analysis.ipynb
└── 03_attention_visualization.ipynb

repos/                    Model repos (git submodules)
├── vggt/
├── dust3r/
├── mast3r/
└── pi3/
```

## Models

| Model | Type | Evaluation metric |
|-------|------|-------------------|
| [VGGT-1B](https://github.com/facebookresearch/vggt) | Multi-view | Depth confidence |
| [DUST3R](https://github.com/naver/dust3r) | Multi-view | Depth confidence |
| [MAST3R](https://github.com/naver/mast3r) | Multi-view | Descriptor confidence |
| [Pi3](https://github.com/jjordanoc/Pi3) | Multi-view | Depth confidence |
| DINOv2-Large | Single-image | Cosine similarity |

All models are evaluated zero-shot using a pairwise encoding strategy: for each trial, we encode all image pairs, extract confidence/similarity scores, and predict the oddity as the image with the lowest mean pairwise score.

## Citation

```bibtex
@article{bonnen2025human,
  title={Human-level 3D shape perception emerges from multi-view learning},
  author={Bonnen, Tyler and Malik, Jitendra and Kanazawa, Angjoo},
  year={2025}
}
```

## License

MIT
