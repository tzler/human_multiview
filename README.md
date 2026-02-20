<div align="center">
<h1>Human-level 3D shape perception emerges from multi-view learning</h1>

<a href="https://tzler.github.io/human_multiview/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://huggingface.co/datasets/tzler/MOCHI"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MOCHI-blue" alt="Dataset"></a>
<a href="https://github.com/tzler/human_multiview"><img src="https://img.shields.io/badge/GitHub-Code-black?logo=github" alt="Code"></a>
<a href="https://arxiv.org/abs/2602.17650"><img src="https://img.shields.io/badge/arXiv-2602.17650-b31b1b" alt="arXiv"></a>


[Tyler Bonnen](https://tzler.github.io/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/)

**UC Berkeley**

</div>

<p align="center">
  <img src="https://tzler.github.io/human_multiview/static/media/main_results.png" width="80%" alt="Main results">
</p>

## Overview

We demonstrate that multi-view transformers are the first vision models to match human performance on 3D shape inference tasks. First we develop a zero-shot evaluation framework (i.e., no task-specific training, fine tuning, or linear decoders) to determine the perceptual abilities of these multi-view models, then we evaluate on the [MOCHI](https://huggingface.co/datasets/tzler/MOCHI) benchmark. [VGGT](https://github.com/facebookresearch/vggt) matches human accuracy, error patterns, and reaction times.


This repository provides the complete analysis pipeline to evaluate models, reproduce results  figures, and explore cross-image attention patterns.

## Quick Start

```bash
# clone with submodules
git clone --recursive https://github.com/tzler/human_multiview.git

# move into the repo and reate conda environment
conda env create -f environment.yml
conda activate human_multiview

# Install this package
pip install -e .
```

## Usage

The repo is a complete pipeline: **scripts generate data, notebooks visualize data**.

### 1. Run model evaluations

```bash
# All models on all MOCHI trials
python scripts/run_evaluation.py --gpu_id 0

# Specific models
python scripts/run_evaluation.py --models vggt dinov2 --gpu_id 0

# Quick test (10 trials)
python scripts/run_evaluation.py --models vggt --n_trials 10 --gpu_id 0
```

### 2. Extract attention data (optional, for Fig 4)

```bash
python scripts/run_attention.py --gpu_id 0
```

### 3. Generate figures

| Notebook | Figures | GPU required |
|----------|---------|:---:|
| `notebooks/01_main_results.ipynb` | Fig 3, S1–S6 | No |
| `notebooks/02_layer_analysis.ipynb` | Fig 2 (bottom), S7 | Yes (for Fig 2) |
| `notebooks/03_attention_visualization.ipynb` | Fig 4, S8, S9 | Yes |

## Models

| Model | Type | Evaluation metric |
|-------|------|-------------------|
| [VGGT-1B](https://github.com/facebookresearch/vggt) | Multi-view | Depth confidence |
| [DUST3R](https://github.com/naver/dust3r) | Multi-view | Depth confidence |
| [MAST3R](https://github.com/naver/mast3r) | Multi-view | Descriptor confidence |
| [Pi3](https://github.com/jjordanoc/Pi3) | Multi-view | Depth confidence |
| DINOv2-Large | Single-image | Cosine similarity |
| VGGT-1B (untrained) | Multi-view | Depth confidence / layer similarity |

All models are evaluated zero-shot using a pairwise encoding strategy: for each trial, we encode all image pairs, extract confidence/similarity scores, and predict the oddity as the image with the lowest mean pairwise score.

An untrained (randomly initialized) VGGT baseline is also available to confirm that performance is attributable to learned representations rather than architecture or input features. It performs at chance on all metrics and layers. See `models/vggt_untrained.py` and run with `--models vggt_untrained --output_dir results/untrained/`.

<details>
<summary>Project structure</summary>

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
│   ├── dinov2.py         DINOv2 baseline
│   └── vggt_untrained.py VGGT-1B (random init)
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

</details>

## Citation

```bibtex
@misc{bonnen2026human,
      title={Human-level 3D shape perception emerges from multi-view learning}, 
      author={Tyler Bonnen and Jitendra Malik and Angjoo Kanazawa},
      year={2026},
      url={https://arxiv.org/abs/2602.17650}, 
}
```

## License

MIT
