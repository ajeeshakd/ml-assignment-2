"""Evaluate the saved assignment models against a labeled test CSV."""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from train_models import CLASS_NAMES, evaluate_model

ROOT = Path(__file__).resolve().parent
METADATA = json.loads((ROOT / "model" / "metadata.json").read_text())


def read_test_data(upload):
    """Validate the assignment's feature columns and binary target labels."""
    columns = next(csv.reader(upload), [])
    if len(columns) != len(set(columns)):
        raise ValueError("CSV column names must be unique.")
    upload.seek(0)
    data = pd.read_csv(upload)
    features = METADATA["features"]
    target = METADATA["target_column"]
    missing = [column for column in [*features, target] if column not in data]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    x = data[features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Features must contain finite numeric values.")
    y = pd.to_numeric(data[target], errors="raise")
    if set(y.unique()) != {0, 1}:
        raise ValueError("ROC AUC requires both target labels: 0 (benign) and 1 (malignant).")
    return x, y.astype(int)


@st.cache_resource
def load_model(relative_path):
    return joblib.load(ROOT / relative_path)


def main():
    st.set_page_config(page_title="Assignment 2 | Model Evaluation", layout="wide")
    st.title("Breast Cancer Classifier Evaluation")
    st.write("Upload a labeled test CSV to evaluate a trained model.")
    st.caption("Target labels: 0 = benign, 1 = malignant. Malignant is the positive class.")
    st.download_button(
        "Download assignment test CSV",
        data=(ROOT / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    models = METADATA["models"]
    selected = st.selectbox("Classifier", list(models), format_func=lambda key: models[key]["display_name"])
    if uploaded is None:
        return

    from io import StringIO
    try:
        x, y = read_test_data(StringIO(uploaded.getvalue().decode("utf-8-sig")))
    except (ValueError, UnicodeError) as exc:
        st.error(f"Check your CSV: {exc}")
        return

    model = load_model(models[selected]["path"])
    metrics, predictions, _ = evaluate_model(model, x, y, score_type=models[selected]["score_type"])
    st.subheader(f"Evaluation on {len(y)} test rows")
    for column, (name, value) in zip(st.columns(len(metrics)), metrics.items()):
        column.metric(name, f"{value:.4f}")

    st.subheader("Confusion matrix")
    counts = confusion_matrix(y, predictions, labels=[0, 1])
    st.dataframe(pd.DataFrame(
        counts,
        index=[f"Actual {name}" for name in CLASS_NAMES],
        columns=[f"Predicted {name}" for name in CLASS_NAMES],
    ))

    with st.expander("Classification report"):
        st.code(classification_report(
            y, predictions, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0,
        ), language="text")


if __name__ == "__main__":
    main()
