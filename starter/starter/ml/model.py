from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_metrics(
    data, model, encoder, lb, categorical_features, label="salary"
):
    """
    Computes model performance metrics on slices of the data for each categorical feature.
    Returns a dictionary with metrics for each slice.
    """
    from starter.ml.data import process_data
    from starter.ml.model import compute_model_metrics, inference

    results = {}
    for feature in categorical_features:
        results[feature] = {}
        for value in data[feature].unique():
            slice_df = data[data[feature] == value]
            if slice_df.shape[0] == 0:
                continue
            X_slice, y_slice, _, _ = process_data(
                slice_df,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            results[feature][value] = {
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
                "count": len(y_slice),
            }
    return results
