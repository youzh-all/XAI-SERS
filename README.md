# XAI-SERS Public Release (Comment 7)

This folder is prepared for public GitHub release to address reviewer comment on data/code availability.

## What is included

- Processed spectra and labels used for the submitted model evaluation:
  - `data/spectra_train.npy` shape `(6255, 331)`
  - `data/spectra_test.npy` shape `(1564, 331)`
  - `data/labels_train.npy` shape `(6255,)`
  - `data/labels_test.npy` shape `(1564,)`
- Trained 1D-CNN model:
  - `model/model_1dcnn.h5`
- Preprocessing and model-training scripts:
  - `code/preprocess_step1_baseline.py`
  - `code/preprocess_step2_despike.py`
  - `code/preprocess_step3_peakbin.py`
  - `code/train_1dcnn.py`
  - `code/train_mlp.py`
  - `code/train_rf.py`
  - `code/train_svm.py`
  - `code/eval_macro_f1.py`
  - `code/explain_shap.py` (single SHAP script kept)

## What is not included

- Raw spectra directories (`data/`, `data_BC/`, `data_BCDS/`, `data_BCDSpBN`) are not included in this compact release folder.
- SHAP result CSV files are intentionally excluded (script is included for regeneration).

## Label index mapping

The label order follows the class list used in the original 1D-CNN script:

- `0`: Shigella sonnei
- `1`: Shigella flexneri
- `2`: Shigella boydii
- `3`: Shigella dysenteriae
- `4`: EIEC
- `5`: EPEC
- `6`: ETEC
- `7`: EAEC
- `8`: STEC

## Original-to-simple filename mapping

- `X_train.npy` -> `data/spectra_train.npy`
- `X_test.npy` -> `data/spectra_test.npy`
- `y_train.npy` -> `data/labels_train.npy`
- `y_test.npy` -> `data/labels_test.npy`
- `model_1DCNN.h5` -> `model/model_1dcnn.h5`
- `model_1DCNN (Feature importance).py` -> `code/train_1dcnn.py`
- `model_MLP.py` -> `code/train_mlp.py`
- `model_RF.py` -> `code/train_rf.py`
- `model_SVM.py` -> `code/train_svm.py`
- `preprocess_BC.py` -> `code/preprocess_step1_baseline.py`
- `preprocesse_BCDS.py` -> `code/preprocess_step2_despike.py`
- `preprocess_BCDSpBN.py` -> `code/preprocess_step3_peakbin.py`
- `comment2_macrof1_table.py` -> `code/eval_macro_f1.py`
- `plot_SHAP-deep.py` -> `code/explain_shap.py`

## Reproducibility note

Scripts were copied from the original working directory with simplified names for readability.  
Some scripts still contain original relative paths and may require path updates depending on your execution location.

## Suggested manuscript statement

"Processed spectra and labels, along with preprocessing/model-training code, are publicly available at our GitHub repository (link to be provided in the final version)."
