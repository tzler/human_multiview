"""Shared figure styling and plotting utilities."""

import matplotlib.pyplot as plt


# Color palette matching the manuscript
COLORS = {
    "human": "#E69F00",       # orange
    "vggt": "#D55E00",        # red-orange
    "dinov2": "#999999",      # grey
    "pi3": "#56B4E9",         # light blue
    "mast3r": "#009E73",      # green
    "dust3r": "#0072B2",      # dark blue
    "match": "#E69F00",       # orange (AA' pairs)
    "nonmatch_ab": "#000000", # black (AB pairs)
    "nonmatch_bp": "#7570B3", # purple (BA' pairs)
}


def set_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (7, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
