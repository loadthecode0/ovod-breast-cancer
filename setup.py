# central_paths.py
import torch

def get_paths_and_device(mode="kaggle"):
    """
    Returns:
        paths: dict containing dataset + weight paths
        device: torch device string ("cuda", "mps", "cpu")
    Default mode = "local" (Mac).
    """

    if mode == "kaggle":
        paths = {
            "test_A": "/kaggle/input/ovod-datasets/dataset_A/dataset_A/test",
            "test_B": "/kaggle/input/ovod-datasets/dataset_B/dataset_B/test",
            "test_C": "/kaggle/input/ovod-datasets/dataset_C/dataset_C/test",
            "train_A": "/kaggle/input/ovod-datasets/dataset_A/dataset_A/train",
            "train_B": "/kaggle/input/ovod-datasets/dataset_B/dataset_B/train",
            "train_C": "/kaggle/input/ovod-datasets/dataset_C/dataset_C/train",
            "weights": "/kaggle/working/weights/",
            "task_2_1_save": "/kaggle/working/task_2_1/",
            "task_2_1_load": "/kaggle/working/task_2_1/",
            "task_2_2_save": "/kaggle/working/task_2_2/",
            "task_2_2_load": "/kaggle/working/task_2_2/",
            "task_3_save": "/kaggle/working/task_3/",
            "task_3_load": "/kaggle/working/task_3/",
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"

    else:  # ===== DEFAULT = LOCAL MAC =====
        paths = {
            "test_A": "./data/dataset_A/test",
            "test_B": "./data/dataset_B/test",
            "test_C": "./data/dataset_C/test",
            "train_A": "./data/dataset_A/train",
            "train_B": "./data/dataset_B/train",
            "train_C": "./data/dataset_C/train",
            "weights": "./weights/",
            "task_2_1_save": "./weights/task_2_1_wts/",
            "task_2_1_load": "./weights/task_2_1_wts/",
            "task_2_2_save": "./weights/task_2_2_wts/",
            "task_2_2_load": "./weights/task_2_2_wts_new/",
            "task_3_save": "./weights/task_3_wts/",
            "task_3_load": "./weights/task_3_wts/",
        }

        # Device selection for Mac (M1/M2/M3)
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    return paths, device, mode
