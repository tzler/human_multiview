"""Attention extraction and cross-image attention visualization for VGGT."""

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .config import N_HEADS


class AttentionExtractor:
    """Extract attention weights from VGGT's global attention blocks.

    Registers forward hooks that recompute Q, K from the input to capture
    attention weights (since VGGT uses fused attention internally).

    Note: RoPE is not applied in the hook, so weights are approximate.
    """

    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self._hooks = []

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x)
            qkv = qkv.reshape(B, N, 3, module.num_heads, module.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if module.q_norm is not None:
                q = module.q_norm(q)
            if module.k_norm is not None:
                k = module.k_norm(k)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            self.attention_maps[layer_idx] = attn.detach().cpu()
        return hook

    def register_hooks(self, layer_indices=None):
        """Register hooks on specified layers."""
        self.clear()
        if layer_indices is None:
            layer_indices = range(len(self.model.aggregator.global_blocks))
        for layer_idx in layer_indices:
            block = self.model.aggregator.global_blocks[layer_idx]
            hook = block.attn.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def clear(self):
        self.attention_maps = {}
        self.remove_hooks()


def extract_attention(vggt_model, img1, img2, extractor, layer_indices=(23,)):
    """Run a forward pass and extract attention weights.

    Args:
        vggt_model: Loaded VGGTModel instance.
        img1, img2: PIL images.
        extractor: AttentionExtractor bound to vggt_model.model.
        layer_indices: Tuple of layer indices to capture.

    Returns:
        attention_maps: Dict layer_idx -> Tensor (1, heads, N, N).
        patch_info: Dict with patch grid dimensions.
    """
    extractor.register_hooks(layer_indices)
    images = vggt_model.preprocess([img1, img2]).unsqueeze(0).to(vggt_model.device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=vggt_model.dtype, enabled="cuda" in vggt_model.device):
            aggregated_tokens_list, patch_start_idx = vggt_model.model.aggregator(images)

    final_layer = aggregated_tokens_list[-1]
    total_tokens = final_layer.shape[2]
    num_patches_per_image = total_tokens - patch_start_idx
    patch_h = patch_w = int(np.sqrt(num_patches_per_image))

    patch_info = {
        "patch_start_idx": patch_start_idx,
        "num_patches_per_image": num_patches_per_image,
        "patch_h": patch_h,
        "patch_w": patch_w,
        "total_tokens": total_tokens,
    }

    attention_maps = extractor.attention_maps.copy()
    extractor.remove_hooks()
    del aggregated_tokens_list, images
    return attention_maps, patch_info


def get_cross_attention(attention_map, patch_info, source_image=0, target_image=1):
    """Extract cross-image patch attention from the full attention matrix.

    In VGGT, tokens from both images are concatenated:
        [img1_special + img1_patches | img2_special + img2_patches]

    Args:
        attention_map: Tensor (1, heads, N, N) where N = 2*total_tokens.
        patch_info: Dict from extract_attention().
        source_image: 0 or 1.
        target_image: 0 or 1.

    Returns:
        Tensor (heads, n_patches, n_patches).
    """
    start_idx = patch_info["patch_start_idx"]
    T = patch_info["total_tokens"]

    if source_image == 0:
        src_start, src_end = start_idx, T
    else:
        src_start, src_end = T + start_idx, 2 * T

    if target_image == 0:
        tgt_start, tgt_end = start_idx, T
    else:
        tgt_start, tgt_end = T + start_idx, 2 * T

    return attention_map[0, :, src_start:src_end, tgt_start:tgt_end]


def pixel_to_patch(x, y, img_size, patch_h, patch_w):
    """Convert pixel coordinates to patch grid indices.

    Args:
        x, y: Pixel coordinates in the original image.
        img_size: (width, height) of the original image.
        patch_h, patch_w: Patch grid dimensions.

    Returns:
        patch_idx: Flattened patch index.
        patch_row, patch_col: Grid coordinates.
    """
    img_w, img_h = img_size
    patch_col = max(0, min(int(x / img_w * patch_w), patch_w - 1))
    patch_row = max(0, min(int(y / img_h * patch_h), patch_h - 1))
    patch_idx = patch_row * patch_w + patch_col
    return patch_idx, patch_row, patch_col


def attention_heatmap(attn_maps, patch_info, src_xy, target_img, layer, head="mean", sigma=1.0):
    """Compute a smoothed attention heatmap for one source point.

    Args:
        attn_maps: Dict from extract_attention().
        patch_info: Dict from extract_attention().
        src_xy: (x, y) pixel coordinates in the source image.
        target_img: PIL image of the target.
        layer: Layer index.
        head: 'mean' to average all heads, or int for a specific head.
        sigma: Gaussian smoothing sigma.

    Returns:
        2D numpy array at target image resolution.
    """
    ph, pw = patch_info["patch_h"], patch_info["patch_w"]
    src_idx, _, _ = pixel_to_patch(src_xy[0], src_xy[1], target_img.size, ph, pw)
    cross = get_cross_attention(attn_maps[layer], patch_info)

    if head == "mean":
        vec = cross[:, src_idx, :].mean(dim=0).numpy()
    else:
        vec = cross[head, src_idx, :].numpy()

    hmap = gaussian_filter(vec.reshape(ph, pw), sigma=sigma)
    return np.array(Image.fromarray(hmap).resize(target_img.size, Image.Resampling.BILINEAR))


def get_object_mask(image, threshold=0.1, fill_holes=False):
    """Generate object mask from luminance thresholding."""
    arr = np.array(image)
    gray = np.mean(arr, axis=2) / 255.0
    mask = gray < (1.0 - threshold)
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=3)
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    return mask


def plot_attention_figure(
    img_A, img_Ap, img_B, points, attn_maps_Ap, attn_maps_B,
    patch_info, layer, head="mean", sigma=1.0, colormap="plasma",
    point_colors=None, include_B=True, mask_attention=True,
):
    """Generate the Fig 4 style multi-panel attention visualization.

    Layout: A (reference with points) | target image | heatmap per point ...
    Rows: A' (match) and optionally B (non-match).

    Args:
        img_A: Source/reference PIL image.
        img_Ap: Matching target PIL image (A').
        img_B: Non-matching target PIL image (B).
        points: List of (x, y) pixel coordinates on img_A.
        attn_maps_Ap: Attention maps dict from extract_attention(A, A').
        attn_maps_B: Attention maps dict from extract_attention(A, B).
        patch_info: Dict from extract_attention().
        layer: Layer index to visualize.
        head: 'mean' or int.
        sigma: Gaussian smoothing.
        colormap: Matplotlib colormap name.
        point_colors: List of color strings. Auto-generated if None.
        include_B: Whether to include the B row.
        mask_attention: Whether to mask attention to the object region.

    Returns:
        matplotlib Figure.
    """
    if point_colors is None:
        point_colors = [
            "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00",
            "#984ea3", "#a65628", "#f781bf", "#999999",
        ]

    targets = [("A'", img_Ap, attn_maps_Ap)]
    if include_B:
        targets.append(("B", img_B, attn_maps_B))

    n_pts = len(points)
    n_rows = len(targets)
    n_data_cols = 1 + n_pts
    col_w, row_h = 3.2, 3.2

    fig = plt.figure(figsize=(col_w * (1 + n_data_cols), row_h * n_rows))
    gs = GridSpec(n_rows, 1 + n_data_cols, figure=fig, wspace=0.02, hspace=0.06)

    # Reference image A with points
    ax_ref = fig.add_subplot(gs[:, 0])
    ax_ref.imshow(img_A)
    for i, (x, y) in enumerate(points):
        ax_ref.plot(x, y, "o", color=point_colors[i % len(point_colors)],
                    markersize=8, markeredgecolor="white", markeredgewidth=1, zorder=5)
    ax_ref.set_title("A", fontsize=15)
    ax_ref.axis("off")

    # Pre-compute heatmaps and masks
    hmaps_all = []
    masks = []
    for _, target_img, attn_maps in targets:
        mask = get_object_mask(target_img) if mask_attention else None
        masks.append(mask)
        row_hmaps = [
            attention_heatmap(attn_maps, patch_info, pt, target_img, layer, head, sigma)
            for pt in points
        ]
        hmaps_all.append(row_hmaps)

    # Column-wise vmax for consistent color scale per point
    col_vmax = []
    for pt_i in range(n_pts):
        vals = []
        for row in range(n_rows):
            if masks[row] is not None:
                masked = np.where(masks[row], hmaps_all[row][pt_i], np.nan)
            else:
                masked = hmaps_all[row][pt_i]
            p99 = np.nanpercentile(masked, 99)
            if not np.isnan(p99):
                vals.append(p99)
        col_vmax.append(max(vals) if vals else max(h[pt_i].max() for h in hmaps_all))

    # Fill grid
    for row, (label, target_img, _) in enumerate(targets):
        # Target image column
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(target_img)
        ax.set_title(label, fontsize=15, y=0.9)
        ax.axis("off")

        # Attention columns
        for pt_i, hmap in enumerate(hmaps_all[row]):
            ax = fig.add_subplot(gs[row, 2 + pt_i])
            if masks[row] is not None:
                masked = np.where(masks[row], hmap, np.nan)
                ax.imshow(target_img, alpha=0.2)
            else:
                masked = hmap
                ax.imshow(target_img)
            ax.imshow(masked, cmap=colormap, alpha=1.0 if mask_attention else 0.7,
                      vmin=0, vmax=col_vmax[pt_i])
            if label == "B":
                ax.set_title("\u25CF", fontsize=15, fontweight="bold",
                             color=point_colors[pt_i % len(point_colors)])
            ax.axis("off")

    return fig
