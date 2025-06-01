"""
Script to compute and save model metrics on data slices
"""
import pandas as pd
import joblib
from starter.ml.model import compute_slice_metrics

def main():
    # Load data and model artifacts
    data = pd.read_csv("starter/data/census_clean.csv")
    model = joblib.load("starter/model/model.joblib")
    encoder = joblib.load("starter/model/encoder.joblib")
    lb = joblib.load("starter/model/lb.joblib")

    # Define categorical features
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

    # Compute slice metrics
    results = compute_slice_metrics(data, model, encoder, lb, cat_features)

    # Write results to file
    with open("slice_output.txt", "w") as f:
        f.write("Model Performance on Data Slices\n")
        f.write("================================\n\n")
        for feature, values in results.items():
            f.write(f"\nFeature: {feature}\n")
            f.write("-" * (len(feature) + 9) + "\n")
            for value, metrics in values.items():
                f.write(f"\nValue: {value}\n")
                f.write(f"Count: {metrics['count']}\n")
                f.write(f"Precision: {metrics['precision']:.3f}\n")
                f.write(f"Recall: {metrics['recall']:.3f}\n")
                f.write(f"F1 (beta=1): {metrics['fbeta']:.3f}\n")
            f.write("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
