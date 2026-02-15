"""VGGT model wrapper for confidence extraction and layer-wise analysis."""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF

from ..config import IMAGE_SIZE, REPOS_DIR, VGGT_MODEL
from .base import MultiViewModel, clear_gpu_memory


class VGGTModel(MultiViewModel):
    """VGGT-1B wrapper for multi-view evaluation.

    Provides:
        - Confidence-based evaluation via depth_conf output.
        - Layer-wise feature extraction for solution layer analysis.
        - DINOv2 encoder feature extraction for baseline comparison.
    """

    def load(self, device="cuda"):
        vggt_path = str(REPOS_DIR / "vggt")
        if vggt_path not in sys.path:
            sys.path.insert(0, vggt_path)
        from vggt.models.vggt import VGGT

        self.device = device
        if "cuda" in device:
            cap = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        self.model = VGGT.from_pretrained(VGGT_MODEL).to(device).eval()

    def preprocess(self, pil_images, target_size=IMAGE_SIZE):
        """Preprocess PIL images for VGGT.

        Resizes to target_size width, rounds height to nearest multiple of 14,
        center-crops if needed.

        Returns:
            Tensor of shape (N, 3, H, W).
        """
        processed = []
        to_tensor = TF.ToTensor()
        for img in pil_images:
            if img.mode != "RGB":
                if img.mode == "RGBA":
                    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    img = Image.alpha_composite(bg, img)
                img = img.convert("RGB")
            w, h = img.size
            new_w = target_size
            new_h = round(h * (new_w / w) / 14) * 14
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            t = to_tensor(img)
            if new_h > target_size:
                start = (new_h - target_size) // 2
                t = t[:, start : start + target_size, :]
            processed.append(t)
        return torch.stack(processed)

    def extract_pairwise_confidence(self, img1, img2):
        """Extract depth_conf maps for an image pair.

        Returns:
            conf_maps: Tensor (1, 2, H, W) on CPU.
        """
        images = self.preprocess([img1, img2]).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=self.dtype, enabled="cuda" in self.device):
                predictions = self.model(images)
        conf = predictions["depth_conf"].detach().cpu()
        del predictions, images
        clear_gpu_memory()
        return conf

    def extract_layer_features(self, img1, img2):
        """Extract patch token features from all 24 aggregator layers.

        Returns:
            aggregated_tokens_list: List of 24 tensors, each (1, 2, T, C) on CPU.
            patch_start_idx: Int, index where patch tokens begin.
        """
        images = self.preprocess([img1, img2]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=self.dtype, enabled="cuda" in self.device):
                aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
        result = [t.detach().cpu() for t in aggregated_tokens_list]
        del aggregated_tokens_list, images
        clear_gpu_memory()
        return result, patch_start_idx

    def extract_dinov2_features(self, img1, img2):
        """Extract features from VGGT's frozen DINOv2-Large encoder.

        Returns:
            tokens_1, tokens_2: Patch tokens (N_patches, C) on CPU.
            cls_1, cls_2: CLS tokens (C,) on CPU.
        """
        images = self.preprocess([img1, img2]).to(self.device)
        encoder = self.model.aggregator.patch_embed
        with torch.no_grad():
            out1 = encoder.forward_features(images[0:1])
            out2 = encoder.forward_features(images[1:2])
        tokens_1 = out1["x_norm_patchtokens"][0].cpu()
        tokens_2 = out2["x_norm_patchtokens"][0].cpu()
        cls_1 = out1["x_norm_clstoken"][0].cpu()
        cls_2 = out2["x_norm_clstoken"][0].cpu()
        del out1, out2, images
        clear_gpu_memory()
        return tokens_1, tokens_2, cls_1, cls_2

    def extract_patch_tokens_at_layer(self, aggregated_tokens, layer_idx, patch_start_idx, image_idx=0):
        """Get patch tokens for a specific layer and image.

        Args:
            aggregated_tokens: List from extract_layer_features().
            layer_idx: Which layer (0-23).
            patch_start_idx: From extract_layer_features().
            image_idx: 0 or 1.

        Returns:
            Tensor (N_patches, C).
        """
        tokens = aggregated_tokens[layer_idx]
        return tokens[0, image_idx, patch_start_idx:, :]
