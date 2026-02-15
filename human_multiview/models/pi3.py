"""Pi3 model wrapper."""

import sys

import torch
import torchvision.transforms as transforms

from ..config import PI3_IMAGE_SIZE, PI3_MODEL, REPOS_DIR
from .base import MultiViewModel, clear_gpu_memory


class Pi3Model(MultiViewModel):
    """Pi3 wrapper for multi-view evaluation."""

    def load(self, device="cuda"):
        pi3_path = str(REPOS_DIR / "pi3")
        if pi3_path not in sys.path:
            sys.path.insert(0, pi3_path)
        from pi3.models.pi3 import Pi3

        self.device = device
        if "cuda" in device:
            cap = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        self.model = Pi3.from_pretrained(PI3_MODEL).to(device).eval()
        clear_gpu_memory()

    def preprocess(self, pil_images, target_size=PI3_IMAGE_SIZE):
        """Preprocess images for Pi3 (resize to 560x560)."""
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ])
        return torch.stack([transform(img.convert("RGB")) for img in pil_images])

    def extract_pairwise_confidence(self, img1, img2):
        """Extract confidence from Pi3 predictions (sigmoid of conf logits).

        Returns:
            conf_maps: Tensor (1, 2, H, W) on CPU.
        """
        imgs_tensor = self.preprocess([img1, img2]).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=self.dtype, enabled="cuda" in self.device):
                results = self.model(imgs_tensor[None])

        conf_maps = torch.sigmoid(results["conf"][0, ..., 0]).detach().cpu()
        conf_maps = conf_maps.unsqueeze(0)  # (1, 2, H, W)

        del results, imgs_tensor
        clear_gpu_memory()
        return conf_maps
