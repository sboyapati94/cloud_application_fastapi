import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X,
    categorical_features=[],
    label=None,
    training=True,
    encoder=None,
    lb=None
):
    """ Process data for machine learning pipeline.

    Processes data using one hot encoding for categorical features and
    label binarizer for labels. Can be used in training or inference.

    Note: Depending on model type, you may want to add functionality
    that scales continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing features and label.
    categorical_features: list[str]
        List of categorical feature names (default=[])
    label : str
        Name of label column in `X`. If None, returns empty array for y.
    training : bool
        True for training mode, False for inference/validation.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training, else returns input encoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training, else returns input binarizer.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(
            sparse=False, handle_unknown="ignore"
        )
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch case where y is None during inference
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
