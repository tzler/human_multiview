"""DINOv2 baseline — uses VGGT's frozen encoder for single-image features."""

import torch
import torch.nn.functional as F

from .vggt import VGGTModel


class DINOv2Model(VGGTModel):
    """DINOv2-Large baseline extracted from VGGT's frozen encoder.

    This is NOT a multi-view model. It extracts single-image features and
    computes pairwise cosine similarity as a baseline.
    """

    def extract_pairwise_confidence(self, img1, img2):
        """Compute cosine similarity between DINOv2 features.

        Returns a pseudo-confidence tensor shaped (1, 2, 1, 1) so the
        evaluation pipeline can treat it uniformly. The value is the
        global-pool cosine similarity (higher = more similar).
        """
        tokens_1, tokens_2, _, _ = self.extract_dinov2_features(img1, img2)
        pool_1 = tokens_1.mean(dim=0)
        pool_2 = tokens_2.mean(dim=0)
        sim = F.cosine_similarity(pool_1.unsqueeze(0), pool_2.unsqueeze(0)).item()
        # Return as (1, 2, 1, 1) — same value for both images in the pair
        conf = torch.tensor([[[[sim]], [[sim]]]])
        return conf

    def extract_all_similarities(self, img1, img2):
        """Compute multiple similarity metrics between DINOv2 features.

        Returns:
            Dict with keys: 'mean', 'max', 'cls', 'global_pool'.
        """
        tokens_1, tokens_2, cls_1, cls_2 = self.extract_dinov2_features(img1, img2)

        t1_norm = F.normalize(tokens_1, p=2, dim=-1)
        t2_norm = F.normalize(tokens_2, p=2, dim=-1)
        sim_matrix = t1_norm @ t2_norm.T

        return {
            "mean": sim_matrix.mean().item(),
            "max": sim_matrix.max().item(),
            "cls": F.cosine_similarity(cls_1.unsqueeze(0), cls_2.unsqueeze(0)).item(),
            "global_pool": F.cosine_similarity(
                tokens_1.mean(dim=0).unsqueeze(0),
                tokens_2.mean(dim=0).unsqueeze(0),
            ).item(),
        }
