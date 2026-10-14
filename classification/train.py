"""Training: baseline model, trial evaluation for the optimisers, and final model.

Commands (run from the project root):
    python -m classification.train --mode baseline   # stage 7: baseline + collapse check
    python -m classification.train --mode final      # stage 10: retrain the selected configuration

Neither mode ever loads the test split (load_splits blocks it).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classification.dataset import (SplitData, compute_class_weights, format_distribution, load_splits,
                                    make_tf_dataset)
from classification.evaluate import compute_metrics, predict_probs
from classification.model import build_from_config
from classification.preprocessing import build_augmentation
from classification.utils import (DISCLAIMER, ProjectError, add_common_args, config_from_args, environment_info,
                                  get_logger, label_info, load_json, path_from_cfg, save_json, set_global_seed)

logger = get_logger(__name__)


class TrainingError(ProjectError):
    """Raised when training is unstable (NaN loss, no learning, missing checkpoint)."""


@dataclass
class TrainResult:
    model: object
    history: dict
    best_epoch: int          # 1-based epoch with the best monitored validation value
    epochs_run: int
    runtime_sec: float
    val_metrics: dict        # keys prefixed with 'val_' (computed on the restored best weights)
    val_probs: np.ndarray


def baseline_params(cfg: dict) -> dict:
    b = cfg["baseline"]
    return {"dense_units": int(b["dense_units"]), "dropout_rate": float(b["dropout_rate"]),
            "learning_rate": float(b["learning_rate"])}


def complete_params(cfg: dict, params: dict) -> dict:
    """Fill hyperparameters that are not being searched with the baseline value."""
    merged = {**baseline_params(cfg), **params}
    return {"dense_units": int(merged["dense_units"]), "dropout_rate": float(merged["dropout_rate"]),
            "learning_rate": float(merged["learning_rate"])}


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def prediction_diagnostics(prob: np.ndarray, y: np.ndarray, threshold: float, min_fraction: float) -> dict:
    """Probability statistics + collapse flag (model predicting (almost) only one class)."""
    prob, y = np.asarray(prob, dtype=float), np.asarray(y, dtype=int)
    pred_pos = float((prob >= threshold).mean())
    has_both = len(np.unique(y)) == 2
    reasons = []
    if has_both and (pred_pos < min_fraction or pred_pos > 1.0 - min_fraction):
        reasons.append(f"predicts Tumor for {100 * pred_pos:.1f}% of validation images although both classes are present")
    if float(prob.std()) < 1e-4:
        reasons.append("output scores are (almost) constant")
    return {
        "score_min": float(prob.min()), "score_mean": float(prob.mean()), "score_max": float(prob.max()),
        "score_std": float(prob.std()),
        "mean_score_true_no_tumor": float(prob[y == 0].mean()) if (y == 0).any() else None,
        "mean_score_true_tumor": float(prob[y == 1].mean()) if (y == 1).any() else None,
        "predicted_tumor_fraction": pred_pos,
        "true_tumor_fraction": float(y.mean()),
        "collapsed": bool(reasons), "collapse_reasons": reasons,
    }


COLLAPSE_CHECKLIST = """
The baseline model collapsed. The optimisation pipeline is STOPPED on purpose. Check, in this order:
  1. Label mapping : labels.class_names = ["No Tumor", "Tumor"]; folders yes->Tumor, no->No Tumor (manifest.csv).
  2. Data loading  : run  python scripts/inspect_dataset.py  and look at artifacts/dataset/sample_images.png.
  3. Class balance : see the class distribution above; try training.use_class_weights=true.
  4. Preprocessing : pixel range must be [0, 1] (the log prints min/max of the first training batch).
  5. Learning rate : try --set baseline.learning_rate=0.0003 (or 0.0001); also check the loss curve.
  6. Probabilities : see score statistics in artifacts/reports/baseline_report.json (all near one value = no learning).
  7. Training time : raise training.max_epochs or lower training.early_stopping_patience pressure.
Fix the cause and re-run the baseline before running any optimisation.
"""


# ----------------------------------------------------------------------------
# Core training function (shared by baseline, search trials, candidates)
# ----------------------------------------------------------------------------
def train_model(params: dict, train: SplitData, val: SplitData, cfg: dict, max_epochs: int, patience: int,
                checkpoint_path: Optional[Path] = None, verbose: int = 0, seed: Optional[int] = None) -> TrainResult:
    """Train one configuration on `train`, monitor `val`, restore the best weights, score `val`."""
    import keras

    seed = cfg["project"]["seed"] if seed is None else seed
    keras.backend.clear_session()
    set_global_seed(seed, cfg["project"].get("deterministic_ops", False))
    params = complete_params(cfg, params)
    tcfg = cfg["training"]

    model = build_from_config(cfg, params)
    augmentation = build_augmentation(cfg["augmentation"], seed)
    train_ds = make_tf_dataset(train.x, train.y, tcfg["batch_size"], training=True, augmentation=augmentation, seed=seed)
    val_ds = make_tf_dataset(val.x, val.y, tcfg["batch_size"], training=False)
    class_weight = compute_class_weights(train.y) if tcfg["use_class_weights"] else None

    callbacks = [
        keras.callbacks.EarlyStopping(monitor=tcfg["monitor"], mode=tcfg["monitor_mode"], patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.TerminateOnNaN(),
    ]
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        callbacks.append(keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor=tcfg["monitor"],
                                                         mode=tcfg["monitor_mode"], save_best_only=True))

    start = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs, class_weight=class_weight, shuffle=False,
                     callbacks=callbacks, verbose=verbose)
    runtime = time.time() - start
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}

    losses = np.array(history["loss"] + history.get("val_loss", []), dtype=float)
    if not np.isfinite(losses).all():
        raise TrainingError("Training diverged: a loss became NaN/inf. Lower the learning rate "
                            "(e.g. --set baseline.learning_rate=0.0003) and check the pixel range of the inputs.")
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise TrainingError(f"No checkpoint was written to {checkpoint_path}; the monitored metric "
                            f"'{tcfg['monitor']}' never improved or was unavailable.")

    monitored = np.array(history[tcfg["monitor"]])
    best_idx = int(np.argmax(monitored) if tcfg["monitor_mode"] == "max" else np.argmin(monitored))
    probs = predict_probs(model, val.x)  # model holds the best weights (restore_best_weights=True)
    metrics = compute_metrics(val.y, probs, tcfg["threshold"])
    val_metrics = {f"val_{k}": v for k, v in metrics.items()}
    val_metrics["val_loss"] = history["val_loss"][best_idx]
    return TrainResult(model=model, history=history, best_epoch=best_idx + 1, epochs_run=len(monitored),
                       runtime_sec=runtime, val_metrics=val_metrics, val_probs=probs)


def make_trial_evaluator(train: SplitData, val: SplitData, cfg: dict) -> Callable[[dict], dict]:
    """Fitness function for ACO / random search. Uses ONLY training and validation data.

    fitness = validation metric optimization.primary_metric (default val_roc_auc) of the best epoch of a
    short proxy training. Trials whose model predicts only one class are returned with status 'collapsed'
    (their real score is logged, but optimisers never reward them).
    """
    import keras

    primary = cfg["optimization"]["primary_metric"]
    search = cfg["optimization"]["search_training"]
    tcfg = cfg["training"]

    def evaluate(params: dict) -> dict:
        result = train_model(params, train, val, cfg, max_epochs=search["max_epochs"],
                             patience=search["early_stopping_patience"])
        if primary not in result.val_metrics:
            raise ProjectError(f"optimization.primary_metric '{primary}' is not one of {sorted(result.val_metrics)}")
        diag = prediction_diagnostics(result.val_probs, val.y, tcfg["threshold"], tcfg["collapse_min_fraction"])
        fitness = float(result.val_metrics[primary])
        out = {"status": "collapsed" if diag["collapsed"] else "ok", "fitness": fitness,
               "best_epoch": result.best_epoch, "epochs_run": result.epochs_run,
               "runtime_sec": round(result.runtime_sec, 2)}
        for key in ("val_roc_auc", "val_accuracy", "val_precision", "val_recall", "val_specificity", "val_f1",
                    "val_loss"):
            out[key] = result.val_metrics[key]
        keras.backend.clear_session()
        return out

    return evaluate


# ----------------------------------------------------------------------------
# Plots / saving
# ----------------------------------------------------------------------------
def plot_history(history: dict, path: Path, title: str) -> None:
    panels = [("loss", "Loss"), ("accuracy", "Accuracy"), ("auc", "AUC (Keras)"), ("recall", "Recall / sensitivity")]
    panels = [p for p in panels if p[0] in history]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False)
    for ax, (key, label) in zip(axes[0], panels):
        epochs = range(1, len(history[key]) + 1)
        ax.plot(epochs, history[key], label="train")
        if f"val_{key}" in history:
            ax.plot(epochs, history[f"val_{key}"], label="validation")
        ax.set_xlabel("epoch")
        ax.set_title(label)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_app_metadata(cfg: dict, params: dict, method: str, trained_on: str, epochs: int) -> Path:
    """Everything the Streamlit app needs: preprocessing, label mapping, threshold, Grad-CAM layer."""
    info = label_info(cfg)
    metadata = {
        "model_file": cfg["app"]["model_file"],
        "class_names": info["class_names"],
        "label_to_index": info["name_to_id"],
        "positive_class": info["positive_class"],
        "score_meaning": "sigmoid output = model score for the positive class (Tumor); not a calibrated probability",
        "threshold": cfg["evaluation"]["threshold"],
        "image": {"size": cfg["image"]["size"], "channels": cfg["image"]["channels"],
                  "resample": cfg["image"]["resample"], "normalization": cfg["image"]["normalization"]},
        "gradcam_layer": cfg["gradcam"]["target_layer"],
        "gradcam_alpha": cfg["gradcam"]["alpha"],
        "selected_method": method, "hyperparameters": params, "trained_on": trained_on, "epochs": epochs,
        "dataset": cfg["dataset"]["name"], "seed": cfg["project"]["seed"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
    }
    return save_json(metadata, path_from_cfg(cfg, "models_dir", create=True) / cfg["app"]["metadata_file"])


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
def run_baseline(cfg: dict) -> int:
    info = label_info(cfg)
    models_dir = path_from_cfg(cfg, "models_dir", create=True)
    reports_dir = path_from_cfg(cfg, "reports_dir", create=True)
    data = load_splits(cfg, ("train", "val"))
    train, val = data["train"], data["val"]

    logger.info("Label mapping: %s | positive class: %s", info["name_to_id"], info["positive_class"])
    logger.info("Train distribution: %s", format_distribution(train.y, info["class_names"]))
    logger.info("Val   distribution: %s", format_distribution(val.y, info["class_names"]))
    logger.info("Input range check (first training image): min=%.3f max=%.3f shape=%s",
                float(train.x[0].min()), float(train.x[0].max()), train.x.shape[1:])
    if train.x.min() < 0 or train.x.max() > 1.0:
        raise TrainingError("Inputs are outside [0, 1]; preprocessing is inconsistent.")
    logger.info("Class weights: %s", compute_class_weights(train.y) if cfg["training"]["use_class_weights"] else "off")

    params = baseline_params(cfg)
    logger.info("Training baseline with %s", params)
    result = train_model(params, train, val, cfg, cfg["training"]["max_epochs"],
                         cfg["training"]["early_stopping_patience"],
                         checkpoint_path=models_dir / "baseline.keras", verbose=2)
    diag = prediction_diagnostics(result.val_probs, val.y, cfg["training"]["threshold"],
                                  cfg["training"]["collapse_min_fraction"])

    pd.DataFrame(result.history).assign(epoch=lambda d: np.arange(1, len(d) + 1)).to_csv(
        reports_dir / "baseline_history.csv", index=False)
    plot_history(result.history, reports_dir / "baseline_curves.png", "Baseline CNN - training vs validation")
    save_json({"method": "baseline", "hyperparameters": params, "best_epoch": result.best_epoch,
               "epochs_run": result.epochs_run, "runtime_sec": result.runtime_sec,
               "val_metrics": result.val_metrics, "diagnostics": diag,
               "label_mapping": info["name_to_id"], "positive_class": info["positive_class"],
               "train_counts": train.counts(), "val_counts": val.counts(),
               "model_file": "baseline.keras", "environment": environment_info(),
               "seed": cfg["project"]["seed"], "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
              reports_dir / "baseline_report.json")

    m = result.val_metrics
    print(f"\nBaseline validation (best epoch {result.best_epoch}/{result.epochs_run}): "
          f"acc={m['val_accuracy']:.4f} precision={m['val_precision']:.4f} recall={m['val_recall']:.4f} "
          f"specificity={m['val_specificity']:.4f} f1={m['val_f1']:.4f} roc_auc={m['val_roc_auc']:.4f}")
    print(f"Score statistics: min={diag['score_min']:.3f} mean={diag['score_mean']:.3f} max={diag['score_max']:.3f}; "
          f"predicted Tumor fraction={diag['predicted_tumor_fraction']:.3f} (true {diag['true_tumor_fraction']:.3f})")
    if diag["collapsed"]:
        print("\nBASELINE COLLAPSED: " + "; ".join(diag["collapse_reasons"]))
        print(COLLAPSE_CHECKLIST)
        return 2
    print("Baseline check passed (both classes are predicted). You may continue with the optimisation step.")
    return 0


def run_final(cfg: dict) -> int:
    import keras

    models_dir = path_from_cfg(cfg, "models_dir", create=True)
    reports_dir = path_from_cfg(cfg, "reports_dir", create=True)
    selected = load_json(reports_dir / "selected_config.json")
    params = complete_params(cfg, selected["params"])
    strategy = cfg["final_training"]["strategy"]
    final_path = models_dir / cfg["app"]["model_file"]
    logger.info("Selected method: %s | params: %s | strategy: %s", selected["method"], params, strategy)

    if strategy == "train_only_best_checkpoint":
        source = models_dir / selected["model_file"]
        if not source.exists():
            raise TrainingError(f"Selected candidate model not found: {source}")
        shutil.copyfile(source, final_path)
        write_app_metadata(cfg, params, selected["method"], "train split only (best validation checkpoint)",
                           selected["best_epoch"])
        print(f"Final model = selected train-only checkpoint, copied to {final_path}")
        return 0
    if strategy != "train_val_fixed_epochs":
        raise ProjectError(f"Unknown final_training.strategy '{strategy}'.")

    data = load_splits(cfg, ("train", "val"))
    dev_x = np.concatenate([data["train"].x, data["val"].x])
    dev_y = np.concatenate([data["train"].y, data["val"].y])
    epochs = int(selected["best_epoch"])
    logger.info("Retraining on train+val (%d images) for %d epochs (best epoch found on validation).",
                len(dev_y), epochs)

    keras.backend.clear_session()
    seed = cfg["project"]["seed"]
    set_global_seed(seed, cfg["project"].get("deterministic_ops", False))
    model = build_from_config(cfg, params)
    ds = make_tf_dataset(dev_x, dev_y, cfg["training"]["batch_size"], training=True,
                         augmentation=build_augmentation(cfg["augmentation"], seed), seed=seed)
    class_weight = compute_class_weights(dev_y) if cfg["training"]["use_class_weights"] else None
    hist = model.fit(ds, epochs=epochs, class_weight=class_weight, verbose=2, shuffle=False,
                     callbacks=[keras.callbacks.TerminateOnNaN()])
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    if not np.isfinite(history["loss"]).all():
        raise TrainingError("Final training diverged (NaN loss).")

    probs = predict_probs(model, dev_x)
    diag = prediction_diagnostics(probs, dev_y, cfg["training"]["threshold"], cfg["training"]["collapse_min_fraction"])
    if diag["collapsed"]:
        raise TrainingError("Final model collapsed on its own training data: " + "; ".join(diag["collapse_reasons"]))
    model.save(final_path)
    write_app_metadata(cfg, params, selected["method"], "train+val", epochs)
    pd.DataFrame(history).assign(epoch=lambda d: np.arange(1, len(d) + 1)).to_csv(
        reports_dir / "final_history.csv", index=False)
    plot_history(history, reports_dir / "final_curves.png", "Final model - training curves (no validation set)")
    save_json({"method": selected["method"], "hyperparameters": params, "epochs": epochs, "strategy": strategy,
               "trained_on": "train+val", "n_images": int(len(dev_y)),
               "train_val_sanity_check": diag, "environment": environment_info(), "seed": seed,
               "note": "Sanity check on training data only; it is NOT a performance estimate. "
                       "The independent estimate comes from classification.evaluate on the test split."},
              reports_dir / "final_config.json")
    print(f"Final model saved to {final_path}\nNext: python -m classification.evaluate")
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the baseline (stage 7) or the final model (stage 10).")
    add_common_args(parser)
    parser.add_argument("--mode", choices=["baseline", "final"], required=True)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    try:
        return run_baseline(cfg) if args.mode == "baseline" else run_final(cfg)
    except ProjectError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
