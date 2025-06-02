# Script to train machine learning model.

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the parent directory to the Python path for relative imports
parent_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(parent_dir, "..")))
from ml.data import process_data  # noqa: E402
from ml.model import train_model  # noqa: E402

# Load the clean data
data = pd.read_csv("starter/data/census_clean.csv")

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

# Split data
train, test = train_test_split(data, test_size=0.20, random_state=42)

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
model = train_model(X_train, y_train)

# Save the model and encoders
model_dir = os.path.join(os.path.dirname(__file__), "../../model")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "model.joblib"))
joblib.dump(encoder, os.path.join(model_dir, "encoder.joblib"))
joblib.dump(lb, os.path.join(model_dir, "lb.joblib"))
