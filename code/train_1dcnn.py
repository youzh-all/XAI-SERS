from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
RESULT_DIR = BASE_DIR / "results_1dcnn"


def parse_args():
    parser = argparse.ArgumentParser(description="Train 1D-CNN with reproducible split and multi-run seeds.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (e.g., 0.70 or 0.80)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio (e.g., 0.15 or 0.10)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (e.g., 0.15 or 0.10)")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for train/val/test split")
    parser.add_argument(
        "--run-seeds",
        type=int,
        nargs="+",
        default=[297, 298, 299, 300, 301],
        help="Seeds for repeated runs (number of runs = len(run_seeds))",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=9e-4, help="Adam learning rate")
    parser.add_argument(
        "--best-by",
        type=str,
        default="test_accuracy",
        choices=["test_accuracy", "test_macro_f1", "val_macro_f1"],
        help="Criterion for selecting/saving the best run model",
    )
    parser.add_argument(
        "--target-test-accuracy",
        type=float,
        default=None,
        help="Stop early once a run reaches this test accuracy threshold",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=RESULT_DIR,
        help="Directory for metrics, config, and plots",
    )
    parser.add_argument("--output-model", type=Path, default=MODEL_DIR / "model_1dcnn.h5")
    return parser.parse_args()


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return gpus


def load_release_data():
    x_train = np.load(DATA_DIR / "spectra_train.npy")
    x_test = np.load(DATA_DIR / "spectra_test.npy")
    y_train = np.load(DATA_DIR / "labels_train.npy")
    y_test = np.load(DATA_DIR / "labels_test.npy")

    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    return x_all, y_all


def split_train_val_test(x: np.ndarray, y: np.ndarray, split_seed: int, ratios: tuple[float, float, float]):
    train_ratio, val_ratio, test_ratio = ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x,
        y,
        test_size=test_ratio,
        random_state=split_seed,
        stratify=y,
    )

    val_from_trainval = val_ratio / (train_ratio + val_ratio)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=val_from_trainval,
        random_state=split_seed,
        stratify=y_trainval,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def build_model(input_dim: int, n_classes: int, lr: float):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=7, activation="relu", input_shape=(input_dim, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(150, activation="relu"))
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def to_cnn_tensor(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray):
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)
    x_train_t = np.expand_dims(x_train_s.astype(np.float32), axis=-1)
    x_val_t = np.expand_dims(x_val_s.astype(np.float32), axis=-1)
    x_test_t = np.expand_dims(x_test_s.astype(np.float32), axis=-1)
    return x_train_t, x_val_t, x_test_t


def metric_row(split: str, y_true: np.ndarray, y_pred: np.ndarray, run_seed: int):
    return {
        "run_seed": run_seed,
        "split": split,
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_confusion_matrix_percent(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm.astype(float), row_sum, where=row_sum != 0) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_percent, cmap=plt.cm.Greens)
    plt.colorbar(im, ax=ax)
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=60, ha="right")
    ax.set_yticklabels(classes)
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            ax.text(j, i, f"{cm_percent[i, j]:.2f}%", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    result_dir = args.result_dir

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    split_ratio_text = f"{args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}"
    gpus = configure_gpu()

    x_all, y_all_raw = load_release_data()
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_all_raw)
    class_names = [str(c) for c in label_encoder.classes_]

    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
        x_all, y_all, split_seed=args.split_seed, ratios=ratios
    )
    x_train_t, x_val_t, x_test_t = to_cnn_tensor(x_train, x_val, x_test)

    result_dir.mkdir(parents=True, exist_ok=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    best_score = -1.0
    best_payload = None

    for run_seed in args.run_seeds:
        set_global_seed(run_seed)

        # Requested settings:
        # epochs=50, early stopping monitor='loss', patience=5, restore_best_weights=True
        callback = EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True,
        )

        model = build_model(input_dim=x_train_t.shape[1], n_classes=len(class_names), lr=args.learning_rate)
        history = model.fit(
            x_train_t,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_val_t, y_val),
            callbacks=[callback],
            verbose=0,
        )

        pred_train = np.argmax(model.predict(x_train_t, verbose=0), axis=1)
        pred_val = np.argmax(model.predict(x_val_t, verbose=0), axis=1)
        pred_test = np.argmax(model.predict(x_test_t, verbose=0), axis=1)

        all_rows.append(metric_row("train", y_train, pred_train, run_seed))
        all_rows.append(metric_row("val", y_val, pred_val, run_seed))
        all_rows.append(metric_row("test", y_test, pred_test, run_seed))

        val_f1 = f1_score(y_val, pred_val, average="macro", zero_division=0)
        test_acc = accuracy_score(y_test, pred_test)
        test_f1_macro = f1_score(y_test, pred_test, average="macro", zero_division=0)

        if args.best_by == "test_accuracy":
            select_score = test_acc
        elif args.best_by == "test_macro_f1":
            select_score = test_f1_macro
        else:
            select_score = val_f1

        if select_score > best_score:
            best_score = select_score
            best_payload = {
                "model": model,
                "run_seed": run_seed,
                "history": history.history,
                "y_test_pred": pred_test,
                "val_macro_f1": val_f1,
                "test_accuracy": test_acc,
                "test_macro_f1": test_f1_macro,
            }

        print(
            f"[run seed={run_seed}] "
            f"val_macro_f1={val_f1:.4f}, "
            f"test_acc={test_acc:.4f}, "
            f"test_macro_f1={test_f1_macro:.4f}"
        )

        if args.target_test_accuracy is not None and test_acc >= args.target_test_accuracy:
            print(
                f"Target reached at run seed={run_seed}: "
                f"test_acc={test_acc:.4f} >= {args.target_test_accuracy:.4f}"
            )
            break

    if best_payload is None:
        raise RuntimeError("No model was trained.")

    best_payload["model"].save(args.output_model)

    df = pd.DataFrame(all_rows)
    df.to_csv(result_dir / "run_metrics_by_split.csv", index=False)

    summary = (
        df.groupby("split")[["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(result_dir / "run_metrics_summary_mean_std.csv", index=False)

    pd.DataFrame(best_payload["history"]).to_csv(result_dir / "best_run_training_history.csv", index=False)
    plot_confusion_matrix_percent(
        y_true=y_test,
        y_pred=best_payload["y_test_pred"],
        classes=class_names,
        out_path=result_dir / "best_run_confusion_matrix_test.png",
    )

    config = pd.DataFrame(
        [
            {"item": "batch_size", "value": args.batch_size},
            {"item": "split_ratio", "value": split_ratio_text},
            {"item": "split_seed", "value": args.split_seed},
            {"item": "number_of_runs", "value": len(args.run_seeds)},
            {"item": "run_seeds", "value": ",".join(str(s) for s in args.run_seeds)},
            {"item": "epochs", "value": args.epochs},
            {"item": "early_stopping_monitor", "value": "loss"},
            {"item": "early_stopping_patience", "value": 5},
            {"item": "early_stopping_restore_best_weights", "value": True},
            {"item": "best_selection_metric", "value": args.best_by},
            {"item": "best_selection_score", "value": best_score},
            {"item": "best_run_seed", "value": best_payload["run_seed"]},
            {"item": "best_run_val_macro_f1", "value": best_payload["val_macro_f1"]},
            {"item": "best_run_test_accuracy", "value": best_payload["test_accuracy"]},
            {"item": "best_run_test_macro_f1", "value": best_payload["test_macro_f1"]},
            {"item": "n_train_samples", "value": len(y_train)},
            {"item": "n_val_samples", "value": len(y_val)},
            {"item": "n_test_samples", "value": len(y_test)},
            {"item": "target_test_accuracy", "value": args.target_test_accuracy},
            {"item": "gpu_devices_visible", "value": len(gpus)},
        ]
    )
    config.to_csv(result_dir / "training_config_used.csv", index=False)

    print("\nSaved artifacts:")
    print(f"- Best model: {args.output_model}")
    print(f"- Run metrics: {result_dir / 'run_metrics_by_split.csv'}")
    print(f"- Summary: {result_dir / 'run_metrics_summary_mean_std.csv'}")
    print(f"- Config: {result_dir / 'training_config_used.csv'}")
    print(f"- Confusion matrix: {result_dir / 'best_run_confusion_matrix_test.png'}")


if __name__ == "__main__":
    main()
