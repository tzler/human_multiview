"""MAST3R model wrapper."""

import sys

import torch

from ..config import MAST3R_MODEL, REPOS_DIR
from .base import MultiViewModel, clear_gpu_memory
from .dust3r import preprocess_for_3r


class MAST3RModel(MultiViewModel):
    """MAST3R wrapper for multi-view evaluation."""

    def load(self, device="cuda"):
        mast3r_path = str(REPOS_DIR / "mast3r")
        if mast3r_path not in sys.path:
            sys.path.insert(0, mast3r_path)
        from mast3r.model import AsymmetricMASt3R

        self.device = device
        self.dtype = torch.float32
        self.model = AsymmetricMASt3R.from_pretrained(MAST3R_MODEL).to(device).eval()
        clear_gpu_memory()

    def extract_pairwise_confidence(self, img1, img2):
        """Extract confidence from MAST3R bidirectional predictions.

        Returns:
            conf_maps: Tensor (1, 2, H, W) on CPU.
        """
        # MAST3R uses DUST3R's inference utilities
        dust3r_path = str(REPOS_DIR / "dust3r")
        if dust3r_path not in sys.path:
            sys.path.insert(0, dust3r_path)
        from dust3r.inference import inference
        from dust3r.image_pairs import make_pairs

        images = preprocess_for_3r([img1, img2])
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

        with torch.no_grad():
            output = inference(pairs, self.model, self.device, batch_size=1)

        # Use descriptor confidence (desc_conf) rather than depth confidence (conf).
        # MASt3R's descriptor confidence is more informative for oddity detection
        # because it reflects feature-matching quality rather than depth uncertainty.
        conf_1 = output["pred1"]["desc_conf"][0].detach().cpu()
        conf_2 = output["pred2"]["desc_conf"][0].detach().cpu()
        conf_maps = torch.stack([conf_1, conf_2]).unsqueeze(0)

        del output, pairs, images
        clear_gpu_memory()
        return conf_maps
