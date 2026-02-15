"""Model wrappers for multi-view evaluation."""

from .base import MultiViewModel
from .vggt import VGGTModel
from .dinov2 import DINOv2Model
from .dust3r import DUST3RModel
from .mast3r import MAST3RModel
from .pi3 import Pi3Model

MODEL_REGISTRY = {
    "vggt": VGGTModel,
    "dust3r": DUST3RModel,
    "mast3r": MAST3RModel,
    "pi3": Pi3Model,
    "dinov2": DINOv2Model,
}


def get_model(name):
    """Get a model class by name.

    Args:
        name: One of 'vggt', 'dust3r', 'mast3r', 'pi3', 'dinov2'.

    Returns:
        Model class (not instantiated).
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
