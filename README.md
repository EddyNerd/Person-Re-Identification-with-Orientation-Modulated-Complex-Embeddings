````markdown
# MEBOW Orientation OT - Clean C4 (Minimum)

Minimal version of the project, ready for GitHub, containing only the core pipeline:
- training (`train_main.py`)
- evaluation (`run_eval_checkpoint.py` / `--eval_only`)
- essential modules (`config`, `data`, `model`, `loss`, `trainer`, `evaluator`, `utils`)
- Hermitian component (`complex_embedding`)
- MEBOW file required by the orientation estimator (`third_party/mebow_official/lib/models/pose_hrnet.py`)

## 1) Requirements

- Python 3.10 or 3.11
- NVIDIA GPU recommended; CPU is possible but slow
- MARS dataset, or an equivalent format

Expected structure:

```text
DATA_ROOT/
  bbox_train/
  bbox_test/
  info/
    train_name.txt
    test_name.txt
    tracks_train_info.mat
    tracks_test_info.mat
    query_IDX.mat
````

## 2) Installation

1. Create the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

2. Install PyTorch according to your machine:
   [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

3. Install the dependencies

```powershell
pip install -r requirements.txt
```

## 3) Required External File: MEBOW Checkpoint

This minimal repository does not include the pre-trained MEBOW weights.

You must place an HBOE checkpoint here:

```text
third_party/mebow_official/models/model_hboe.pth
```

Without this file, modes that use `film_orientation_source="stripe_estimator"` will fail.

## 4) Main Commands

EXP5 C4 training:

```powershell
python train_main.py --root D:\datasets\MARS --preset EXP5_FiLM_Hermitian
```

Evaluation only:

```powershell
python train_main.py --root D:\datasets\MARS --preset EXP5_FiLM_Hermitian --eval_only --checkpoint runs_EXP5_FiLM_Hermitian_C4/best_model.pth
```

## 5) Preserved Content: Strict Minimum

* `README.md`
* `requirements.txt`
* `.gitignore`
* `config_optimized.py`
* `data_optimized.py`
* `evaluator.py`
* `loss_optimized.py`
* `model_optimized.py`
* `run_eval_checkpoint.py`
* `train_main.py`
* `trainer.py`
* `utils_optimized.py`
* `complex_embedding/`
* `third_party/mebow_official/lib/models/pose_hrnet.py`

## 6) Notes

* The main preset remains `EXP5_FiLM_Hermitian`, with `C_stripes=4` and `R_rows=4`.
* Demo scripts, figures, analyses, and heavy artifacts were intentionally removed.

```
```
