# MEBOW Orientation OT - Clean C4 (minimum)

Version minimale du projet, prete pour GitHub, contenant uniquement le pipeline coeur:
- entrainement (`train_main.py`)
- evaluation (`run_eval_checkpoint.py` / `--eval_only`)
- modules essentiels (`config`, `data`, `model`, `loss`, `trainer`, `evaluator`, `utils`)
- composant hermitien (`complex_embedding`)
- fichier MEBOW requis par l'estimateur d'orientation (`third_party/mebow_official/lib/models/pose_hrnet.py`)

## 1) Prerequis

- Python 3.10 ou 3.11
- GPU NVIDIA recommande (CPU possible mais lent)
- Dataset MARS (ou format equivalent)

Structure attendue:

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
```

## 2) Installation

1. Creer l'environnement

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

2. Installer PyTorch (selon ta machine):
https://pytorch.org/get-started/locally/

3. Installer les dependances

```powershell
pip install -r requirements.txt
```

## 3) Fichier externe requis (MEBOW checkpoint)

Ce depot minimal ne contient pas les poids pre-entraines MEBOW.

Tu dois placer un checkpoint HBOE ici:

```text
third_party/mebow_official/models/model_hboe.pth
```

Sans ce fichier, les modes qui utilisent `film_orientation_source="stripe_estimator"` echoueront.

## 4) Commandes principales

Entrainement EXP5 C4:

```powershell
python train_main.py --root D:\datasets\MARS --preset EXP5_FiLM_Hermitian
```

Evaluation seule:

```powershell
python train_main.py --root D:\datasets\MARS --preset EXP5_FiLM_Hermitian --eval_only --checkpoint runs_EXP5_FiLM_Hermitian_C4/best_model.pth
```

## 5) Contenu conserve (strict minimum)

- `README.md`
- `requirements.txt`
- `.gitignore`
- `config_optimized.py`
- `data_optimized.py`
- `evaluator.py`
- `loss_optimized.py`
- `model_optimized.py`
- `run_eval_checkpoint.py`
- `train_main.py`
- `trainer.py`
- `utils_optimized.py`
- `complex_embedding/`
- `third_party/mebow_official/lib/models/pose_hrnet.py`

## 6) Notes

- Le preset principal reste `EXP5_FiLM_Hermitian` avec `C_stripes=4` et `R_rows=4`.
- Les scripts de demo, figures, analyses et artefacts lourds ont ete retires volontairement.
