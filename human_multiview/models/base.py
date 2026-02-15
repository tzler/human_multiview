"""Abstract base class for model evaluation."""

import gc
from abc import ABC, abstractmethod

import torch


class MultiViewModel(ABC):
    """Base class for multi-view model wrappers.

    Each model must implement:
        - load(): Load model weights onto device.
        - extract_pairwise_confidence(img1, img2): Return confidence maps for a pair.
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.dtype = None

    @abstractmethod
    def load(self, device="cuda"):
        """Load model onto device. Sets self.model, self.device, self.dtype."""

    @abstractmethod
    def extract_pairwise_confidence(self, img1, img2):
        """Extract confidence maps for an image pair.

        Args:
            img1, img2: PIL images.

        Returns:
            conf_maps: Tensor of shape (1, 2, H, W) — confidence for each image
                       in the pair, or equivalent structure.
        """

    def unload(self):
        """Free model from GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        clear_gpu_memory()

    @property
    def name(self):
        return self.__class__.__name__.replace("Model", "").lower()


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
