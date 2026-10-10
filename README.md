# Brain MRI Binary Classification (Tumor vs No Tumor)

An independent **educational / research-portfolio** project: a custom CNN for binary brain-MRI classification,
a transparent **Ant Colony Optimization (ACO)-inspired** hyperparameter search compared against **random search**
and an untuned **baseline**, Grad-CAM interpretability, and a Streamlit demo app.

> ## ⚠️ Medical disclaimer
> This is an **educational research prototype. It is NOT a medical device and NOT a diagnostic tool.**
> It has not been clinically validated, was trained on a small public dataset, and has had no external validation.
> Never use it to diagnose, screen for, rule out, or make decisions about any medical condition or treatment.
> Always consult a qualified healthcare professional.

**Status of results:** no results are included in this repository. All numbers appear only after you run the
pipeline yourself; the tables in [Results](#results) are empty templates.

---

## 1. Project objective (Stage 1 - problem definition)

| Item | Definition |
|---|---|
| Task | Binary image classification of single 2-D brain MRI slices |
| Input | One brain MRI image (JPG/PNG), resized to 128 x 128, grayscale, pixels / 255 |
| Output | A sigmoid score in [0, 1] = model score for **Tumor**; label **Tumor** if score >= 0.5, else **No Tumor** |
| Positive class | **Tumor** (label id 1). **No Tumor** = label id 0 |

**Research questions**

1. Can a small custom CNN separate Tumor from No Tumor MRI slices on a held-out test set?
2. Does an ACO-inspired search over dense units, dropout and learning rate find a configuration with better
   *validation* ROC-AUC than a hand-picked baseline and than random search with the same proposal budget?
3. How large are the differences compared with run-to-run noise?

**Measurable objectives**

- Report accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC and the confusion matrix on an
  untouched test set (with bootstrap 95% intervals).
- Compare baseline vs ACO vs random search on identical splits, preprocessing, metrics and proposal budget, and
  report the budget (proposals, unique trainings, training time).
- Publish the outcome whatever it is (baseline better / no difference / ACO better).

**Not in scope:** clinical validation, tumour sub-typing, segmentation, localisation, multi-centre generalisation.

## 2. Methodology overview

The sequence of stages follows the general workflow of ACO-tuned CNN studies (dataset -> preprocessing ->
augmentation -> CNN -> pheromone-guided hyperparameter search -> final training -> evaluation), adapted to two classes.
This project uses a different dataset, different classes, a different architecture and its own implementation.

| Stage | What happens | Where |
|---|---|---|
| 2 Dataset inspection | counts, sizes, corrupt files, duplicates, sample images | `scripts/inspect_dataset.py` |
| 3 Splitting | 80/10/10 stratified, duplicate-aware, manifest saved | `scripts/create_splits.py` |
| 4 Preprocessing | grayscale, 128 x 128 bilinear, /255, same code in the app | `classification/preprocessing.py` |
| 5 Augmentation | training batches only: +/-10 deg rotation, 5% shift, 10% zoom (no flips) | `classification/preprocessing.py` |
| 6 Baseline CNN | 7 conv layers + GAP + dense + dropout + sigmoid | `classification/model.py` |
| 7 Baseline training | early stopping on val AUC, checkpoint, **collapse check** | `classification/train.py` |
| 8 ACO search | pheromone-guided search, full CSV log | `optimization/aco_optimizer.py` |
| 9 Comparison | baseline vs ACO vs random search, **validation data only** | `optimization/compare_optimizers.py` |
| 10 Final training | selected config retrained on train+val for the validated epoch count | `classification/train.py --mode final` |
| 11 Final evaluation | test set evaluated **once** | `classification/evaluate.py` |
| 12 Grad-CAM | heatmaps from the last conv feature map | `classification/gradcam.py` |
| 13 Web app | Streamlit upload -> score -> Grad-CAM | `app.py` |

## 3. Dataset

**Br35H :: Brain Tumor Detection 2020** (Kaggle upload by Ahmed Hamada)

- Kaggle: <https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection> (slug `ahmedhamada0/brain-tumor-detection`)
- Original Br35H record: IEEE DataPort, DOI `10.21227/tbkk-q937`
- Classes: folders `yes/` (tumorous MRI) -> **Tumor**, `no/` (non-tumorous MRI) -> **No Tumor**
- The Kaggle description lists 1,500 images in each of the two folders. A different IEEE DataPort page describes an
  801-image annotated version of Br35H, so published counts differ between sources. **The counts produced by
  `scripts/inspect_dataset.py` on your downloaded copy are the authoritative ones.** The download may also contain an
  unlabelled extra folder (e.g. `pred/`); the code ignores any folder that is not mapped to a class and reports it.
- **License / usage:** Kaggle lists "Data files (c) Original Authors" (no open license). Use it for non-commercial
  education/research, cite the source, and **do not redistribute the images** (they are git-ignored).
- **Not used:** the 4-class Kaggle "Brain Tumor Classification (MRI)" dataset. The code refuses folders that look like
  it (`glioma_tumor`, `meningioma_tumor`, `pituitary_tumor`).
- **No patient identifiers** are provided, so patient-level separation between splits **cannot be guaranteed**.
  Near-duplicate detection reduces, but does not remove, the risk that slices of the same patient end up in different
  splits, which can make results look better than they would on new patients.
- No official test split exists in the yes/no folders, so the project creates its own (below).
- If you use another binary dataset, edit `dataset.tumor_folders` / `dataset.no_tumor_folders` in the config; folders of
  tumour sub-types (glioma, meningioma, pituitary) are merged into **Tumor**.

### Dataset preparation

**Option A - Kaggle API (automatic, opt-in).** Create a token at Kaggle -> Settings -> API -> *Create New Token*
(downloads `kaggle.json`). Install `kaggle`, make the token available, then:

```bash
pip install kaggle
python scripts/inspect_dataset.py --download
```

**Option B - manual (always works).** Download the zip in your browser from the Kaggle link, unzip it so that
`data/raw/br35h/yes` and `data/raw/br35h/no` exist (or use any folder and pass `--data-dir`).

## 4. Folder structure

```
brain-tumor-binary-classification/
├── classification/
│   ├── __init__.py
│   ├── utils.py            # config, logging, seeding, JSON helpers (shared)
│   ├── dataset.py          # discovery, inspection, duplicates, splitting, loading
│   ├── preprocessing.py    # image loading, normalisation, augmentation
│   ├── model.py            # CNN + compile
│   ├── train.py            # baseline / final training, trial evaluator, collapse check
│   ├── evaluate.py         # metrics, plots, one-time test evaluation
│   └── gradcam.py          # Grad-CAM
├── optimization/
│   ├── __init__.py
│   ├── common.py           # shared trial runner (cache, failure handling, logging)
│   ├── aco_optimizer.py    # ACO-inspired search
│   ├── random_search.py    # random-search baseline
│   └── compare_optimizers.py
├── scripts/
│   ├── inspect_dataset.py
│   ├── create_splits.py
│   └── verify_setup.py
├── configs/config.yaml
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated at run time (git-ignored, except the final model and its metadata): `artifacts/{dataset,splits,models,optimization,reports}/`.
`classification/utils.py` and `optimization/common.py` are the only additions to the requested structure; they hold code
used by several modules.

## 5. Environment setup

Python 3.10-3.12, TensorFlow >= 2.16 (Keras 3). A GPU is optional; everything runs on CPU, just slower.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/verify_setup.py --skip-data             # environment + code smoke test
```

All commands are run from the project root.

## 6. Training and experiment commands

```bash
# 1. Inspect the data (tables + figures in artifacts/dataset/)
python scripts/inspect_dataset.py                       # add --download for Kaggle API download

# 2. Create the leakage-safe split manifest (artifacts/splits/manifest.csv)
python scripts/create_splits.py

# 3. Check everything incl. data and manifest
python scripts/verify_setup.py

# 4. Baseline + collapse diagnosis      (exit code 2 and a checklist if the model predicts one class)
python -m classification.train --mode baseline

# 5. ACO search, random search, candidate training, validation comparison, selection
python -m optimization.compare_optimizers
#    (the two searches can also be run alone: python -m optimization.aco_optimizer
#     and python -m optimization.random_search; then add --reuse-search to the comparison)

# 6. Retrain the selected configuration (train+val, epoch count from validation)
python -m classification.train --mode final

# 7. Final, one-time evaluation on the untouched test set
python -m classification.evaluate

# 8. Web app
streamlit run app.py
```

Dependencies between commands: 2 needs the data; 4 needs 2; 5 needs 4 (it stops if the baseline collapsed);
6 needs `reports/selected_config.json` from 5; 7 needs 6 (and the three candidate models from 4-5);
8 needs `artifacts/models/final_model.keras` + `app_metadata.json` from 6.

Any config value can be overridden, e.g. `--set training.max_epochs=5 --set optimization.aco.n_ants=4`.
Use `--data-dir /path/to/folder` if the images are elsewhere (e.g. `/kaggle/input/...`).

## 7. ACO search procedure

Implemented in `optimization/aco_optimizer.py`. This is an **ACO-inspired search adapted to a small discrete space**;
the design decisions below are my own choices and the code makes no claim to reproduce any published algorithm exactly.

- **Representation.** A solution is `{dense_units, dropout_rate, learning_rate}`. Each parameter is a decision layer;
  each allowed value is a node. Default space: dense units {32, 64, 128}, dropout {0.2, 0.3, 0.4, 0.5},
  learning rate {3e-4, 1e-3, 3e-3} = 36 configurations (delete the learning-rate line to tune only the first two = 12).
- **Pheromone initialisation.** `tau = 1.0` on every node (unbiased first iteration).
- **Candidate selection** (per parameter, ACS rule): with probability `q0 = 0.2` take the node with the highest
  pheromone (exploitation); otherwise sample node `v` with probability `tau_v^alpha / sum(tau^alpha)` (exploration).
- **Fitness.** Validation ROC-AUC of a short proxy training (<= 6 epochs, early stopping), computed on the restored
  best weights. Only training and validation data are used; `load_splits` blocks the test split for all optimiser code.
- **Pheromone update** after each iteration of 10 ants:
  `tau <- (1 - rho) tau` (rho = 0.3), then every valid ant adds `Q * s_k` to the nodes it used, where
  `s_k = (f_k - f_min) / (f_max - f_min)` over that iteration's valid ants (0.5 if all are equal); the best configuration so far
  gets an extra `Q * 0.5`; finally `tau` is clipped to `[0.1, 5.0]`. Min-max scaling is used because validation AUCs are
  usually very close, so raw values would give nearly equal deposits.
- **Failures.** A trial that raises an exception, returns NaN, or whose model predicts (almost) only one class is logged
  with its status (`failed` / `collapsed`) and error message, receives **no fitness-based reward and no pheromone deposit**,
  and can never be the "best" configuration. No score is ever substituted.
- **Budget and fairness.** 10 ants x 5 iterations = 50 proposals; random search gets the same 50 proposals. Identical
  configurations are trained once (`cache_evaluations`) and repeats are logged with `cached = true`, so the **number of unique
  trainings** (the real cost) is reported separately for each method, with training time. Both methods use the same
  fitness function, seeds, data and logging code.
- **Caveat.** With a space this small, 50 random proposals already cover a large share of all configurations, so random
  search is a strong competitor and the two methods may tie. Training noise (single seed) can be as large as the
  differences between configurations.
- **Outputs** in `artifacts/optimization/`: `aco_search_log.csv`, `random_search_log.csv` (every trial and its validation
  metrics), `aco_pheromone_history.json`, `aco_best.json`, `random_search_best.json`.

## 8. Evaluation

- **Selection (validation only):** highest `val_roc_auc` among candidates that do not collapse; ties -> lower validation
  loss. Written to `artifacts/reports/selected_config.json` (also stores the spread between candidates and a caution
  when it is below 0.01).
- **Final training:** the selected configuration is retrained on train+val for exactly the best epoch found on validation
  (no validation monitoring, no test data). Alternative: `final_training.strategy: train_only_best_checkpoint`.
- **Test evaluation:** `python -m classification.evaluate` evaluates the final model plus the three train-only candidates
  once and writes a lock file; re-running requires `--allow-repeat` and is recorded. The decision threshold is fixed at 0.5.
- **Definitions** (positive class = Tumor): sensitivity = TP/(TP+FN); specificity = TN/(TN+FP); precision = TP/(TP+FP);
  F1 = 2PR/(P+R). Metrics with a zero denominator are reported as NaN, not 0.
- **Outputs** in `artifacts/reports/test/`: `test_summary.csv`, `test_report_final.json` (with bootstrap 95% intervals),
  `classification_report_final.txt`, `confusion_matrix_final.png`, `roc_curves.png`, `pr_curves.png`,
  `test_predictions_final.csv`, `misclassified_examples.png` (contains MRI images - for your own error analysis; do not publish).
- **Class imbalance:** balanced class weights in the loss, and sensitivity/specificity/F1/AUC reported next to accuracy.

### Results

*Not computed yet. Fill these tables from `artifacts/reports/comparison_validation.csv` and
`artifacts/reports/test/test_summary.csv` after running the pipeline. Report them as they are, including negative results.*

| Method | Val ROC-AUC | Test accuracy | Sensitivity | Specificity | F1 | Test ROC-AUC | Unique trainings | Search time |
|---|---|---|---|---|---|---|---|---|
| Baseline | _not run_ | _not run_ | _not run_ | _not run_ | _not run_ | _not run_ | - | - |
| ACO-tuned | _not run_ | _not run_ | _not run_ | _not run_ | _not run_ | _not run_ | _not run_ | _not run_ |
| Random search | _not run_ | _not run_ | _not run_ | _not run_ | _not run_ | _not run_ | _not run_ | _not run_ |

To show figures in the README, copy chosen plots (not MRI images) to `docs/figures/`.

## 9. Streamlit app and deployment

```bash
streamlit run app.py
```

The app loads `artifacts/models/final_model.keras` and `app_metadata.json` (preprocessing, label mapping, threshold,
Grad-CAM layer are read from the metadata, so inference uses exactly the training preprocessing). It never trains,
never starts the ACO search and never downloads data. If the model files are missing it shows an error and stops.
The score is the sigmoid output for *Tumor*; it is **not** a calibrated probability or a diagnosis. Grad-CAM shows
where the model's score came from, not clinical reasoning.

**Deployment**

- *Streamlit Community Cloud / Hugging Face Spaces (Streamlit SDK)*: push the repo (the `.gitignore` allows only
  `final_model.keras` and `app_metadata.json` from `artifacts/`; no MRI images). Set the main file to `app.py`.
  If the host is small, use `tensorflow-cpu` in `requirements.txt`. Keep the disclaimer visible.
- *Vercel* is not suitable: it is built for serverless functions/static sites, while Streamlit needs a long-running
  server and TensorFlow exceeds typical serverless size limits.
- Check the dataset terms before publishing a model trained on it, and never upload patient data to a public demo.

### Google Colab / Kaggle quick path

```python
# Colab: upload the project folder (or git clone your repo), then:
%cd brain-tumor-binary-classification
!pip install -q -r requirements.txt          # Colab already has TensorFlow; this only adds the missing packages
# Kaggle API token (optional): upload kaggle.json, then
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json && pip install -q kaggle
!python scripts/inspect_dataset.py --download     # or upload/unzip manually into data/raw/br35h
!python scripts/create_splits.py
!python -m classification.train --mode baseline
!python -m optimization.compare_optimizers
!python -m classification.train --mode final
!python -m classification.evaluate
```
In a Kaggle notebook, add the dataset through *Add data* and run the scripts with
`--data-dir /kaggle/input/<dataset-folder>`; write outputs under `/kaggle/working` by copying the project there.
Enable a GPU runtime to speed up training (not required).

## 10. Reproducibility settings

- Master seed 42 (`project.seed`) for python, numpy and Keras; split seed 42 (`split.seed`); the manifest
  (`artifacts/splits/manifest.csv`) stores every file path with its label and split, so the split is reproducible.
- Every search trial uses the same initialisation seed, so differences between trials come from the hyperparameters;
  a single seed still leaves training noise. For stronger claims repeat with several seeds.
- Bit-exact repeatability is not guaranteed across hardware/TensorFlow versions (set `project.deterministic_ops: true`
  for deterministic ops, which is slower and may fail on some GPU ops).
- `artifacts/reports/*_report.json` store versions, device information, hyperparameters and seeds.

## 11. Limitations

- Small public dataset; slices are 2-D images, not full 3-D studies; MRI sequence and scanner details are unknown.
- No patient IDs: slices of one patient may appear in several splits (only duplicates/near-duplicates are controlled),
  which can inflate test scores. Possible duplicates between this dataset and other public datasets were not checked.
- Class balance of the downloaded copy must be checked from your own `dataset_summary.csv`; class weights are used regardless.
- A single dataset, single hold-out test set, single training seed, **no external validation**.
- The ACO is a compact adaptation tuned on a short proxy training; the search space is small, so ACO may not beat
  random search or the baseline.
- Grad-CAM is a model-attention visualisation only. Scores are uncalibrated. The model cannot tell non-brain images from brain MRI.
- Image-level classification of cropped, pre-processed slices does not reflect clinical workflows.

## 12. Ethical and medical-use disclaimer

This code is for learning and portfolio purposes. Do not use it for diagnosis, triage, screening or treatment decisions.
Do not upload identifiable patient images to any public deployment. Respect the dataset's terms; do not redistribute the images.
No claims of clinical reliability, novelty or superiority are made; any performance statement must come from your own
logged results.

## 13. Pre-publication checklist

- [ ] No MRI images, zips or `kaggle.json` in the repo (`git ls-files | grep -E "\.(jpg|png|zip|json)$"`).
- [ ] README results table filled from real files, or clearly marked "not run".
- [ ] Dataset source, license note and the "no patient IDs" limitation are stated.
- [ ] Disclaimer visible in README and app screenshots.
- [ ] Reference to prior work is cited only with the author's permission if it is unpublished.
- [ ] `python scripts/verify_setup.py` passes on a clean environment.
