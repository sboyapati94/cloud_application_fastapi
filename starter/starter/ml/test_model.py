import numpy as np
import pandas as pd
import joblib
from starter.starter.ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    compute_slice_metrics,
)


def test_train_model():
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    model = train_model(X, y)
    assert hasattr(model, "predict"), "Model should have a predict method."
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_compute_model_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_inference():
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])
    model = train_model(X, y)
    preds = inference(model, X)
    assert np.array_equal(preds, model.predict(X))
    assert preds.shape[0] == X.shape[0]


def test_compute_slice_metrics():
    # Load data and artifacts
    data = pd.read_csv("starter/data/census_clean.csv")
    model = joblib.load("starter/model/model.joblib")
    encoder = joblib.load("starter/model/encoder.joblib")
    lb = joblib.load("starter/model/lb.joblib")
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    results = compute_slice_metrics(
        data, model, encoder, lb, cat_features, label="salary"
    )
    # Check that results are returned for at least one feature and value
    assert isinstance(results, dict)
    found = False
    for feature in cat_features:
        if feature in results and results[feature]:
            found = True
            break
    assert found, "No slice metrics computed."
