"""Core evaluation logic: oddity prediction, confidence margins, solution layers."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.transform import resize as sk_resize


def get_object_mask(image, threshold=0.1):
    """Generate binary object mask from a PIL image.

    Assumes objects are darker than a white background.

    Args:
        image: PIL image.
        threshold: Luminance threshold (0-1). Pixels darker than
                   (1-threshold) are considered object.

    Returns:
        Binary numpy mask (H, W).
    """
    arr = np.array(image)
    gray = np.mean(arr, axis=2) / 255.0
    mask = gray < (1.0 - threshold)
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=3)
    return mask


def extract_confidence_with_metric(conf_map, mask=None, metric="mean"):
    """Aggregate a confidence map into a single scalar.

    Args:
        conf_map: Tensor or array. Handles shapes (B,S,H,W), (S,H,W), (H,W).
        mask: Optional binary mask (H,W). If provided, only masked pixels count.
        metric: 'mean', 'median', or 'max'.

    Returns:
        float: Aggregated confidence score.
    """
    if isinstance(conf_map, torch.Tensor):
        conf_map = conf_map.cpu().numpy()
    if conf_map.ndim == 4:
        conf_map = conf_map[0, 0]
    elif conf_map.ndim == 3:
        conf_map = conf_map[0]

    if mask is not None:
        if mask.shape != conf_map.shape:
            mask = sk_resize(mask, conf_map.shape, order=0, preserve_range=True).astype(bool)
        values = conf_map[mask]
    else:
        values = conf_map.flatten()

    if len(values) == 0:
        return 0.0

    if metric == "mean":
        return float(np.mean(values))
    elif metric == "median":
        return float(np.median(values))
    elif metric == "max":
        return float(np.max(values))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def predict_oddity(pairwise_scores, n_images):
    """Predict the oddity from pairwise scores.

    The oddity is the image with the lowest mean pairwise score.

    Args:
        pairwise_scores: Dict mapping (i,j) tuples to float scores.
        n_images: Number of images in the trial (typically 3 or 4).

    Returns:
        predicted_oddity: Index of predicted odd image.
        image_scores: Dict mapping image index to mean score.
    """
    image_scores = {}
    for img_idx in range(n_images):
        scores = [
            score
            for (i, j), score in pairwise_scores.items()
            if img_idx == i or img_idx == j
        ]
        image_scores[img_idx] = np.mean(scores) if scores else 0.0
    predicted_oddity = min(image_scores, key=image_scores.get)
    return predicted_oddity, image_scores


def compute_confidence_margin(image_scores, predicted_oddity, n_images):
    """Compute the confidence margin for the oddity prediction.

    margin = second_lowest_score - lowest_score

    A higher margin means the model is more confident in distinguishing
    the predicted oddity from the other images.

    Args:
        image_scores: Dict from predict_oddity, mapping image idx to mean score.
        predicted_oddity: Index of the predicted odd image.
        n_images: Number of images.

    Returns:
        float: Confidence margin (gap between lowest and second-lowest scores).
    """
    sorted_scores = sorted(image_scores.values())
    return float(sorted_scores[1] - sorted_scores[0])


def evaluate_trial_confidence(model, trial_images, metrics=None, masking_options=None):
    """Evaluate a trial using confidence-based oddity prediction.

    Runs all pairwise image combinations through the model, extracts confidence,
    and predicts the oddity for each metric/masking combination.

    Args:
        model: A MultiViewModel instance (loaded).
        trial_images: List of PIL images.
        metrics: List of aggregation metrics. Default: ['mean', 'median', 'max'].
        masking_options: List of bools. Default: [True, False].

    Returns:
        Dict mapping metric_key (e.g. 'mean_masked') to result dict with
        keys: 'predicted_oddity', 'correct', 'image_scores', 'confidence_margin'.
    """
    if metrics is None:
        metrics = ["mean", "median", "max"]
    if masking_options is None:
        masking_options = [True, False]

    n_images = len(trial_images)

    # Initialize score storage
    scores = {}
    for metric in metrics:
        for use_mask in masking_options:
            key = f"{metric}_{'masked' if use_mask else 'unmasked'}"
            scores[key] = {}

    # Compute pairwise confidence
    for i in range(n_images):
        for j in range(i + 1, n_images):
            conf_maps = model.extract_pairwise_confidence(trial_images[i], trial_images[j])
            mask_i = get_object_mask(trial_images[i])
            mask_j = get_object_mask(trial_images[j])

            for metric in metrics:
                for use_mask in masking_options:
                    key = f"{metric}_{'masked' if use_mask else 'unmasked'}"
                    m_i = mask_i if use_mask else None
                    m_j = mask_j if use_mask else None
                    conf_i = extract_confidence_with_metric(conf_maps[:, 0:1], m_i, metric)
                    conf_j = extract_confidence_with_metric(conf_maps[:, 1:2], m_j, metric)
                    scores[key][(i, j)] = (conf_i + conf_j) / 2.0

            del conf_maps

    # Predict oddity for each metric
    results = {}
    for key, pairwise_scores in scores.items():
        pred, image_scores = predict_oddity(pairwise_scores, n_images)
        margin = compute_confidence_margin(image_scores, pred, n_images)
        results[key] = {
            "predicted_oddity": pred,
            "image_scores": image_scores,
            "confidence_margin": margin,
        }

    return results


def compute_similarity(tokens_1, tokens_2, metric="mean"):
    """Compute similarity between two sets of patch tokens.

    Args:
        tokens_1, tokens_2: Tensors of shape (N_patches, C).
        metric: 'mean', 'max', or 'global_pool'.

    Returns:
        float: Similarity score.
    """
    t1 = F.normalize(tokens_1.float(), p=2, dim=-1)
    t2 = F.normalize(tokens_2.float(), p=2, dim=-1)

    if metric == "global_pool":
        return F.cosine_similarity(t1.mean(dim=0).unsqueeze(0), t2.mean(dim=0).unsqueeze(0)).item()
    sim_matrix = t1 @ t2.T
    if metric == "mean":
        return sim_matrix.mean().item()
    elif metric == "max":
        return sim_matrix.max().item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def evaluate_trial_layers(vggt_model, trial_images, ground_truth,
                          similarity_metrics=None, layers=None):
    """Evaluate a trial using layer-wise similarity (VGGT-specific).

    For each layer, computes pairwise similarity and predicts the oddity.
    Also computes the earliest reliable (solution) layer.

    Args:
        vggt_model: Loaded VGGTModel instance.
        trial_images: List of PIL images.
        ground_truth: Correct oddity index.
        similarity_metrics: List of metrics. Default: ['mean', 'max', 'global_pool'].
        layers: List of layer indices. Default: range(24).

    Returns:
        Dict with:
            'per_layer': nested dict [metric][layer] -> bool (correct)
            'solution_layers': dict [metric] -> int or None
    """
    if similarity_metrics is None:
        similarity_metrics = ["mean", "max", "global_pool"]
    if layers is None:
        layers = list(range(24))

    n_images = len(trial_images)

    # Extract layer features for all pairs
    pair_features = {}
    for i in range(n_images):
        for j in range(i + 1, n_images):
            agg_tokens, patch_start_idx = vggt_model.extract_layer_features(
                trial_images[i], trial_images[j]
            )
            pair_features[(i, j)] = (agg_tokens, patch_start_idx)

    # Evaluate each layer
    per_layer = {m: {} for m in similarity_metrics}

    for layer_idx in layers:
        for metric in similarity_metrics:
            pairwise_sims = {}
            for (i, j), (agg_tokens, psi) in pair_features.items():
                t1 = vggt_model.extract_patch_tokens_at_layer(agg_tokens, layer_idx, psi, 0)
                t2 = vggt_model.extract_patch_tokens_at_layer(agg_tokens, layer_idx, psi, 1)
                pairwise_sims[(i, j)] = compute_similarity(t1, t2, metric)

            pred, _ = predict_oddity(pairwise_sims, n_images)
            per_layer[metric][layer_idx] = pred == ground_truth

    # Find solution layers
    solution_layers = {}
    for metric in similarity_metrics:
        solution_layers[metric] = find_solution_layer(per_layer[metric], layers)

    del pair_features
    return {"per_layer": per_layer, "solution_layers": solution_layers}


def evaluate_trial_dinov2(dinov2_model, trial_images, ground_truth):
    """Evaluate a trial using DINOv2 features (baseline).

    Args:
        dinov2_model: Loaded DINOv2Model instance.
        trial_images: List of PIL images.
        ground_truth: Correct oddity index.

    Returns:
        Dict mapping metric name to bool (correct).
    """
    n_images = len(trial_images)
    metric_names = ["mean", "max", "cls", "global_pool"]
    all_sims = {m: {} for m in metric_names}

    for i in range(n_images):
        for j in range(i + 1, n_images):
            sims = dinov2_model.extract_all_similarities(trial_images[i], trial_images[j])
            for m in metric_names:
                all_sims[m][(i, j)] = sims[m]

    results = {}
    for m in metric_names:
        pred, _ = predict_oddity(all_sims[m], n_images)
        results[m] = pred == ground_truth

    return results


def find_solution_layer(per_layer_correct, layers=None):
    """Find the earliest layer where the model is correct and stays correct.

    Args:
        per_layer_correct: Dict mapping layer index to bool (correct).
        layers: Ordered list of layer indices to check.

    Returns:
        int or None: Earliest solution layer, or None if never stable.
    """
    if layers is None:
        layers = sorted(per_layer_correct.keys())

    for start_idx, start_layer in enumerate(layers):
        all_subsequent_correct = all(
            per_layer_correct.get(layer, False) for layer in layers[start_idx:]
        )
        if all_subsequent_correct:
            return start_layer
    return None
