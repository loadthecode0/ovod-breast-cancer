## Project Overview

This repository contains scripts to train and evaluate GroundingDINO-based detectors for tumor localization using:
- Zero-shot prompts (Task 1)
- CoOp (Context Optimization) training and evaluation (Task 2.1)
- CoCoOp (Conditional CoOp) training and evaluation (Task 2.2)
- Semi-supervised CoOp training and evaluation (Task 3)

Paths, device, and runtime mode are centrally configured via `setup.get_paths_and_device()`.


### Environment and setup

- Python 3.9+ recommended.
- Suggested packages (derived from imports in the scripts):
  - torch, torchvision, numpy, tqdm, matplotlib, supervision, opencv-python, pandas (for CSVs in `utils_a2.py`), groundingdino (provided via the `local_gdino` folder).
- Ensure the repo-local GroundingDINO overlay is on `sys.path` as done in the scripts:
  - `sys.path.insert(0, "local_gdino")`
- Configure your dataset and weights directories in `setup.py` so `get_paths_and_device()` returns correct `PATHS`, `DEVICE`, and `MODE` for your machine.
- Many scripts auto-adjust for Kaggle (`MODE == "kaggle"`) by appending a Kaggle utils path.


### Datasets and annotations

- Scripts expect image directories referred to in `PATHS` (e.g., `PATHS["train_A"]`, `PATHS["test_B"]`, etc.).
- Annotation CSVs are loaded relative to the dataset directory (e.g., `os.path.join(dataset_path, csv_name)`).
- Common CSV names used:
  - Train: `train.csv` or `train_updated.csv`
  - Test: `test.csv` or `test_updated.csv`


## Scripts and CLI usage

Below, each script’s CLI is documented based on its `main(...)` signature. Defaults come from `setup.get_paths_and_device()` and `utils_a2.py` (`DEFAULT_PROMPTS`, `DEFAULT_THRESHOLDS`) unless otherwise noted.


### Task 1: Zero-shot evaluation

File: `task_1.py`

Main:
- `main(dataset_path=None, csv_path=None, prompt=None, max_examples=None)`

CLI (positional):
1. `dataset_path` (optional): Directory containing images.
2. `csv_path` (optional): CSV file inside `dataset_path` (e.g., `test.csv`).
3. `prompt` (optional): Text prompt (e.g., `"a malignant tumor"`).
4. `max_examples` (optional int): Limits number of examples.

Behavior:
- If no arguments are provided, it evaluates three test sets using defaults:
  - Datasets: `PATHS["test_A"]`, `PATHS["test_B"]`, `PATHS["test_C"]`
  - CSVs: `["test.csv", "test_updated.csv", "test.csv"]`
  - Prompts: `DEFAULT_PROMPTS["A"][0]`, `DEFAULT_PROMPTS["B"][0]`, `DEFAULT_PROMPTS["C"][0]`
- Uses `DEFAULT_THRESHOLDS["box_threshold"]` and `DEFAULT_THRESHOLDS["text_threshold"]`.

Examples:
```bash
# Default three-dataset evaluation
python task_1.py

# Single dataset with CSV and custom prompt
python task_1.py /abs/path/to/test_A test.csv "a malignant tumor" 200
```


### Task 2.1: CoOp training

File: `train_task_2_1.py`

Main:
- `main(dataset_path=None, csv_path=None, save_path=None, num_examples=None)`

CLI (positional):
1. `dataset_path` (optional): Train image directory (default `PATHS["train_B"]`).
2. `csv_path` (optional): Train CSV inside the dataset (default `train_updated.csv`).
3. `save_path` (optional): Output `.pt` file; by default auto-derived as `PATHS["task_2_1_save"] + <dataset_id>.pt`
4. `num_examples` (optional int): Subsample training set.

Training details (in-script defaults):
- `init_prompt = DEFAULT_PROMPTS["B"][0]`
- `lr = 5e-2`, `epochs = 30`, `k_context = 8`

Examples:
```bash
# Default B split training with defaults
python train_task_2_1.py

# Custom dataset and CSV, with output path and a 5k-sample subset
python train_task_2_1.py /abs/path/to/train_B train_updated.csv /abs/path/to/out/task2_1.pt 5000
```


### Task 2.2: CoCoOp training

File: `train_task_2_2.py`

Main:
- `main(dataset_path=None, csv_path=None, save_path=None, num_examples=None)`

CLI (positional):
1. `dataset_path` (optional): Train image directory (defaults to `PATHS["train_B"]` if used).
2. `csv_path` (optional): Train CSV (default `train_updated.csv` if used).
3. `save_path` (optional): Output `.pt` file (default auto-derived under `PATHS["task_2_2_save"]` if used).
4. `num_examples` (optional int): Subsample training set.

Important note:
- The `__main__` section currently calls `main(...)` three times to train on A/B/C splits sequentially with built-in defaults, and ignores the parsed dataset/csv/save args. If you want single-run CLI behavior, adapt the bottom of the file to call `main(dataset_path, csv_path, save_path, num_examples)` once.

Training details (in-script defaults):
- `init_prompt = DEFAULT_PROMPTS["B"][0]`
- `lr = 5e-2`, `epochs = 30`, `k_context = 8`

Examples:
```bash
# As-is, runs three trainings (A, B, C) using defaults
python train_task_2_2.py

# If adapted for single-run CLI, an example would look like:
python train_task_2_2.py /abs/path/to/train_A train.csv /abs/path/to/out/task2_2.pt 10000
```


### Task 3: Semi-supervised CoOp training

File: `train_task_3.py`

Main:
- `main(labeled_path=None, labeled_csv=None, unlabeled_path=None, unlabeled_csv=None, save_path=None, num_examples=None, lambda_u=1.0)`

CLI (intended mapping by main signature):
1. `labeled_path` (optional): Labeled image directory (default `PATHS["train_B"]`).
2. `labeled_csv` (optional): Labeled CSV (default `train_updated.csv`).
3. `unlabeled_path` (optional): Unlabeled image directory (default `PATHS["train_C"]`).
4. `unlabeled_csv` (optional): Unlabeled CSV (default `train.csv`).
5. `save_path` (optional): Output `.pt` (default auto-derived under `PATHS["task_3_save"]` using dataset IDs and `lambda_u`).
6. `num_examples` (optional int): Subsample labeled data.
7. `lambda_u` (optional float): Weight for unsupervised loss (default `1.0`).

Training details (in-script defaults):
- `init_prompt = DEFAULT_PROMPTS["B"][0]`
- `lr = 5e-2`, `epochs = 30`, `k_context = 8`
- Unsupervised thresholding and augmentations are defined in the script; `lambda_u` controls unsupervised loss weight.

Examples:
```bash
# Default semi-supervised setup (B labeled, C unlabeled)
python train_task_3.py

# Custom labeled/unlabeled datasets, output path, with 2k labeled samples and lambda_u=0.5
python train_task_3.py /abs/path/to/train_B train_updated.csv /abs/path/to/train_C train.csv /abs/path/to/out/task3.pt 2000 0.5
```


### Task 2.1: CoOp evaluation

File: `test_task_2_1.py`

Main:
- `main(dataset_path=None, csv_path=None, context_path=None, prompt=None, max_examples=None)`

CLI (positional):
1. `dataset_path` (optional): Test image directory (default `PATHS["test_C"]`).
2. `csv_path` (optional): Test CSV (default `test.csv`).
3. `context_path` (optional): Path to trained CoOp context `.pt`. Default in-script: `PATHS["task_2_1_load"] + "_data_dataset_C_train.pt"`.
4. `prompt` (optional): Text prompt (default `DEFAULT_PROMPTS["A"][0]`).
5. `max_examples` (optional int): Limit number of examples.

Examples:
```bash
# Evaluate with default paths and context
python test_task_2_1.py

# Evaluate a custom dataset using a specific trained context
python test_task_2_1.py /abs/path/to/test_C test.csv /abs/path/to/contexts/B_train.pt "a malignant tumor" 500
```


### Task 2.2: CoCoOp evaluation

File: `test_task_2_2.py`

Main:
- `main(dataset_path=None, csv_path=None, context_path=None, prompt=None, max_examples=None)`

CLI (positional):
1. `dataset_path` (optional): Test image directory (default `PATHS["test_B"]`).
2. `csv_path` (optional): Test CSV (default `test_updated.csv`).
3. `context_path` (optional): Path to trained CoCoOp context `.pt`. Default in-script: `PATHS["task_2_2_load"] + "_data_dataset_B_train.pt"`.
4. `prompt` (optional): Text prompt (default `DEFAULT_PROMPTS["A"][0]`).
5. `max_examples` (optional int): Limit number of examples.

Note:
- The evaluator also loads a MetaNet from `PATHS["task_2_2_load"] + "_data_dataset_B_train_meta_net.pt"`.

Examples:
```bash
# Evaluate with defaults (B split)
python test_task_2_2.py

# Custom dataset and context
python test_task_2_2.py /abs/path/to/test_B test_updated.csv /abs/path/to/contexts/cocoop_B.pt "a malignant tumor" 400
```


### Task 3: Semi-supervised CoOp evaluation

File: `test_task_3.py`

Main:
- `main(dataset_path=None, csv_path=None, context_path=None, prompt=None, max_examples=None)`

CLI (positional):
1. `dataset_path` (optional): Test image directory (default `PATHS["test_C"]`).
2. `csv_path` (optional): Test CSV (default `test.csv`).
3. `context_path` (optional): Path to trained semi-supervised CoOp context `.pt`. Default in-script: `PATHS["task_3_load"] + "A_train_C_train_1_0.pt"`.
4. `prompt` (optional): Text prompt (default `DEFAULT_PROMPTS["A"][0]`).
5. `max_examples` (optional int): Limit number of examples.

Examples:
```bash
# Evaluate with default semi-supervised context
python test_task_3.py

# Evaluate with custom context and prompt
python test_task_3.py /abs/path/to/test_C test.csv /abs/path/to/contexts/A_train_C_train_1_0.pt "a dense tumor lump" 300
```