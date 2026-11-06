"""Learning reference: train and evaluate the five Assignment 2 classifiers.

Run in BITS Virtual Lab: python train_models.py
Jupyter equivalent:     %run train_models.py

Read the accompanying guide and understand/adapt the experiment before submission.
The script trains once; the later Streamlit app should only load saved pipelines.
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # Works in a lab terminal without a graphical display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SEED = 42
CLASS_NAMES = ["benign", "malignant"]
DATASET_URL = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"


def load_dataset():
    """Load scikit-learn's bundled copy of the UCI dataset; no download needed."""
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    # sklearn originally encodes malignant=0, benign=1. Reverse deliberately:
    # here target=1 means malignant, the positive class for precision/recall/F1.
    target = (dataset.target == 0).astype(int).rename("target")
    if features.shape[0] < 500 or features.shape[1] < 12:
        raise ValueError("Dataset does not meet the assignment minimum size.")
    if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Unexpected missing or non-finite feature values.")
    return features, target, dataset.DESCR


def build_models():
    """Fixed baseline settings; do not tune them against the held-out test data."""
    return {
        "logistic_regression": (
            "Logistic Regression",
            Pipeline([("scaler", StandardScaler()),
                      ("classifier", LogisticRegression(max_iter=5000, random_state=SEED))]),
        ),
        "decision_tree": (
            "Decision Tree",
            Pipeline([("classifier", DecisionTreeClassifier(random_state=SEED))]),
        ),
        "knn": (
            "K-Nearest Neighbors",
            Pipeline([("scaler", StandardScaler()),
                      ("classifier", KNeighborsClassifier(n_neighbors=5))]),
        ),
        "naive_bayes": (
            "Gaussian Naive Bayes",
            Pipeline([("classifier", GaussianNB())]),
        ),
        "svm": (
            "Support Vector Machine",
            Pipeline([("scaler", StandardScaler()),
                      ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))]),
        ),
    }


def evaluate_model(model, features, target, *, score_type=None):
    """AUC uses continuous scores, never the thresholded predictions."""
    predictions = model.predict(features)
    if score_type is None:
        classifier = model.named_steps["classifier"]
        score_type = "decision_function" if isinstance(classifier, SVC) and not classifier.probability else "probability"
    if score_type == "probability":
        positive_column = list(model.classes_).index(1)
        scores = model.predict_proba(features)[:, positive_column]
    else:
        # Binary SVC decision scores increase toward classes_[1], which is 1.
        if list(model.classes_) != [0, 1]:
            raise ValueError("Expected class order [0, 1] for binary decision scores.")
        scores = model.decision_function(features)
    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, scores),
        "Precision": precision_score(target, predictions, pos_label=1, zero_division=0),
        "Recall": recall_score(target, predictions, pos_label=1, zero_division=0),
        "F1": f1_score(target, predictions, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }
    return metrics, predictions, scores


def main(output_dir):
    root = Path(output_dir).resolve()
    model_dir = root / "model"
    results_dir = root / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    features, target, description = load_dataset()
    # Split raw observations FIRST. Each pipeline fits its scaler on training
    # data only, including within each cross-validation fold.
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, stratify=target, random_state=SEED
    )
    for name, x_part, y_part in [
        ("train_data.csv", x_train, y_train),
        ("test_data.csv", x_test, y_test),
    ]:
        x_part.join(y_part).to_csv(root / name, index=False)
    (results_dir / "dataset_description.txt").write_text(description, encoding="utf-8")

    lines = []

    def log(message=""):
        print(message, flush=True)
        lines.append(str(message))

    log("MACHINE LEARNING ASSIGNMENT 2 - FIVE CLASSIFIERS")
    log(f"Execution UTC: {datetime.now(timezone.utc).isoformat()}")
    log("Dataset: UCI Breast Cancer Wisconsin (Diagnostic)")
    log(f"Samples: {len(features)} | Input features: {features.shape[1]}")
    log(f"Training rows: {len(x_train)} | Held-out test rows: {len(x_test)}")
    log("Target: 0 = benign; 1 = malignant (positive class)")
    log("Training-only five-fold CV F1 is used to select a baseline model.")
    log("All six required metrics are evaluated on the SAME held-out test split.")

    models = build_models()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    # Select using training data only, before inspecting any test metrics.
    for slug, (name, pipeline) in models.items():
        scores = cross_val_score(pipeline, x_train, y_train, cv=folds, scoring="f1")
        cv_rows.append({"Model": name, "CV F1 mean": float(scores.mean()),
                        "CV F1 std": float(scores.std())})
    cv_table = pd.DataFrame(cv_rows)
    selected = cv_table.loc[cv_table["CV F1 mean"].idxmax(), "Model"]
    cv_table.to_csv(results_dir / "training_cv_scores.csv", index=False)
    log(f"Selected by training CV F1 (first listed if tied): {selected}")

    rows, manifest = [], {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (slug, (name, pipeline)) in zip(axes.flat, models.items()):
        pipeline.fit(x_train, y_train)
        metrics, predictions, scores = evaluate_model(pipeline, x_test, y_test)
        rows.append({"Model": name, **metrics})

        # Persist the ENTIRE fitted pipeline, so inference uses the same scaler.
        saved_path = model_dir / f"{slug}.joblib"
        joblib.dump(pipeline, saved_path)
        restored = joblib.load(saved_path)
        restored_metrics, restored_predictions, _ = evaluate_model(restored, x_test, y_test)
        np.testing.assert_array_equal(predictions, restored_predictions)
        np.testing.assert_allclose(list(metrics.values()), list(restored_metrics.values()))

        report = classification_report(
            y_test, predictions, labels=[0, 1], target_names=CLASS_NAMES,
            digits=4, zero_division=0,
        )
        (results_dir / f"{slug}_classification_report.txt").write_text(report, encoding="utf-8")
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        pd.DataFrame(cm, index=["actual_benign", "actual_malignant"],
                     columns=["predicted_benign", "predicted_malignant"]).to_csv(
                         results_dir / f"{slug}_confusion_matrix.csv")
        pd.DataFrame({"source_row_index": x_test.index, "actual": y_test.to_numpy(),
                      "predicted": predictions, "positive_score": scores}).to_csv(
                          results_dir / f"{slug}_predictions.csv", index=False)
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
        manifest[slug] = {"display_name": name, "path": f"model/{slug}.joblib",
                          "score_type": "decision_function" if isinstance(pipeline.named_steps["classifier"], SVC)
                          and not pipeline.named_steps["classifier"].probability else "probability"}
        log(f"Saved and reload-verified: {name}")

    axes.flat[-1].axis("off")
    fig.suptitle("Held-out test confusion matrices (positive class: malignant)")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "model_comparison.csv", index=False)

    packages = ["scikit-learn", "numpy", "pandas", "matplotlib", "joblib", "scipy"]
    versions = {package: version(package) for package in packages}
    (root / "requirements-trained.txt").write_text(
        "\n".join(f"{name}=={value}" for name, value in versions.items()) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic)", "source_url": DATASET_URL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": features.columns.tolist(), "target_column": "target",
        "class_mapping": {"0": "benign", "1": "malignant"}, "positive_class": 1,
        "random_state": SEED, "test_size": 0.20,
        "training_rows": len(x_train), "test_rows": len(x_test),
        "training_row_indices": x_train.index.tolist(), "test_row_indices": x_test.index.tolist(),
        "selection_method": "Highest mean training-only five-fold stratified CV F1",
        "selected_model": selected, "models": manifest,
        "python_version": platform.python_version(), "package_versions": versions,
        "note": "Execution location is not verified by this script. Capture your real BITS Lab screenshot.",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log("\nHELD-OUT TEST METRICS")
    log(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    log("\nAll five saved pipelines passed reload verification.")
    log(f"Output folder: {root}")
    log("If running in BITS Virtual Lab, capture the actual lab screen with this table.")
    log("Keep model/, test_data.csv, results/, and requirements-trained.txt for the next step.")
    (results_dir / "execution_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent),
                        help="Destination for models, CSVs, and results (default: script folder).")
    args = parser.parse_args()
    main(args.output_dir)
