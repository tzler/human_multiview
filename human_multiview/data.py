"""MOCHI dataset loading and trial helpers."""

from datasets import load_dataset

from .config import MOCHI_DATASET


def load_mochi():
    """Load the MOCHI dataset from HuggingFace.

    Returns:
        dataset: HuggingFace dataset with all trials.
    """
    return load_dataset(MOCHI_DATASET)["train"]


def get_trial_images(trial):
    """Parse a MOCHI trial into canonical (A, A', B) ordering.

    Args:
        trial: A single trial dict from the MOCHI dataset.

    Returns:
        img_A: First image of the matching object.
        img_A_prime: Second image of the matching object (different viewpoint).
        img_B: Image of the oddity (different object).
        trial_info: Dict with trial metadata.
    """
    images = trial["images"]
    oddity_idx = trial["oddity_index"]

    img_B = images[oddity_idx]
    same_indices = [i for i in range(len(images)) if i != oddity_idx]
    img_A = images[same_indices[0]]
    img_A_prime = images[same_indices[1]] if len(same_indices) > 1 else images[same_indices[0]]

    trial_info = {
        "trial_name": trial["trial"],
        "dataset": trial["dataset"],
        "condition": trial.get("condition", trial["dataset"]),
        "oddity_index": oddity_idx,
        "n_images": len(images),
        "human_accuracy": trial.get("human_avg", None),
        "human_rt": trial.get("RT_avg", None),
    }

    return img_A, img_A_prime, img_B, trial_info


def trials_by_condition(dataset):
    """Group trial indices by condition.

    Args:
        dataset: MOCHI HuggingFace dataset.

    Returns:
        Dict mapping condition name to list of trial indices.
    """
    groups = {}
    for i in range(len(dataset)):
        condition = dataset[i]["dataset"]
        if condition not in groups:
            groups[condition] = []
        groups[condition].append(i)
    return groups
