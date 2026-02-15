"""DUST3R model wrapper."""

import sys

import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image
from PIL.ImageOps import exif_transpose

from ..config import DUST3R_MODEL, REPOS_DIR, THREE_R_IMAGE_SIZE, THREE_R_PATCH_SIZE
from .base import MultiViewModel, clear_gpu_memory

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def preprocess_for_3r(pil_images, size=THREE_R_IMAGE_SIZE, patch_size=THREE_R_PATCH_SIZE):
    """Preprocess images for the 3R model family (DUST3R, MAST3R).

    Resizes to fit within `size`, then crops to patch-aligned dimensions.

    Returns:
        List of dicts with 'img' tensor, 'true_shape', 'idx', 'instance'.
    """
    imgs = []
    for img in pil_images:
        img = exif_transpose(img).convert("RGB")
        S = max(img.size)
        interp = Image.LANCZOS if S > size else Image.BICUBIC
        new_size = tuple(int(round(x * size / S)) for x in img.size)
        img = img.resize(new_size, interp)
        W, H = img.size
        cx, cy = W // 2, H // 2
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )
    return imgs


class DUST3RModel(MultiViewModel):
    """DUST3R wrapper for multi-view evaluation."""

    def load(self, device="cuda"):
        dust3r_path = str(REPOS_DIR / "dust3r")
        if dust3r_path not in sys.path:
            sys.path.insert(0, dust3r_path)
        from dust3r.model import AsymmetricCroCo3DStereo

        self.device = device
        self.dtype = torch.float32  # DUST3R uses float32
        self.model = AsymmetricCroCo3DStereo.from_pretrained(DUST3R_MODEL).to(device).eval()
        clear_gpu_memory()

    def extract_pairwise_confidence(self, img1, img2):
        """Extract confidence from DUST3R bidirectional predictions.

        Returns:
            conf_maps: Tensor (1, 2, H, W) on CPU.
        """
        dust3r_path = str(REPOS_DIR / "dust3r")
        if dust3r_path not in sys.path:
            sys.path.insert(0, dust3r_path)
        from dust3r.inference import inference
        from dust3r.image_pairs import make_pairs

        images = preprocess_for_3r([img1, img2])
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

        with torch.no_grad():
            output = inference(pairs, self.model, self.device, batch_size=1)

        conf_1 = output["pred1"]["conf"][0].detach().cpu()
        conf_2 = output["pred2"]["conf"][0].detach().cpu()
        conf_maps = torch.stack([conf_1, conf_2]).unsqueeze(0)

        del output, pairs, images
        clear_gpu_memory()
        return conf_maps
