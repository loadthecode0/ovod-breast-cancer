# central_paths.py
import torch

def get_paths_and_device(mode="local"):
    """
    Returns:
        paths: dict containing dataset + weight paths
        device: torch device string ("cuda", "mps", "cpu")
    Default mode = "local" (Mac).
    """

    if mode == "kaggle":
        paths = {
            "dataset_A": "/kaggle/input/ovod-datasets/dataset_A/dataset_A/test",
            "dataset_B": "/kaggle/input/ovod-datasets/dataset_B/dataset_B/test",
            "dataset_C": "/kaggle/input/ovod-datasets/dataset_C/dataset_C/test",
            "weights": "/kaggle/working/weights/"
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"

    else:  # ===== DEFAULT = LOCAL MAC =====
        paths = {
            "dataset_A": "./data/dataset_A/test",
            "dataset_B": "./data/dataset_B/test",
            "dataset_C": "./data/dataset_C/test",
            "weights": "./weights/"
        }

        # Device selection for Mac (M1/M2/M3)
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    return paths, device, mode
