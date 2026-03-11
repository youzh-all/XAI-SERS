import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_BCDSpBN"
OUTPUT_DIR = BASE_DIR / "outputs_BCDSpBN"
CNN_MODEL_PATH = OUTPUT_DIR / "model_1DCNN.h5"
OUT_TABLE_PATH = OUTPUT_DIR / "table2_metrics_with_macro_f1.csv"


def load_dataset():
    X_train = np.load(DATASET_DIR / "X_train.npy")
    X_test = np.load(DATASET_DIR / "X_test.npy")
    y_train = np.load(DATASET_DIR / "y_train.npy")
    y_test = np.load(DATASET_DIR / "y_test.npy")
    return X_train, X_test, y_train, y_test


def metric_row(model_name, y_true, y_pred):
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_test": int(len(y_true)),
    }


def eval_1dcnn_saved(X_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_test_cnn = np.expand_dims(X_test_s.astype(np.float32), axis=-1)

    model = load_model(CNN_MODEL_PATH)
    y_prob = model.predict(X_test_cnn, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred


def eval_mlp(X_train, X_test, y_train):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)
    return clf.predict(X_test_s)


def eval_rf(X_train, X_test, y_train):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train_s, y_train)
    return clf.predict(X_test_s)


def eval_svm(X_train, X_test, y_train):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = svm.SVC(kernel="linear")
    clf.fit(X_train_s, y_train)
    return clf.predict(X_test_s)


def main():
    X_train, X_test, y_train, y_test = load_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    y_pred_cnn = eval_1dcnn_saved(X_train, X_test, y_test)
    rows.append(metric_row("1D-CNN (saved)", y_test, y_pred_cnn))

    y_pred_mlp = eval_mlp(X_train, X_test, y_train)
    rows.append(metric_row("MLP", y_test, y_pred_mlp))

    y_pred_rf = eval_rf(X_train, X_test, y_train)
    rows.append(metric_row("RF", y_test, y_pred_rf))

    y_pred_svm = eval_svm(X_train, X_test, y_train)
    rows.append(metric_row("SVM", y_test, y_pred_svm))

    df = pd.DataFrame(rows)
    df["f1_gap_weighted_minus_macro"] = df["f1_weighted"] - df["f1_macro"]
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_TABLE_PATH, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved: {OUT_TABLE_PATH}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
