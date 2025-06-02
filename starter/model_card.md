# Model Card

## Model Details

This model is a RandomForestClassifier implemented using scikit-learn. It predicts whether an individual's income exceeds $50K/year based on demographic and employment features from the UCI Census Income dataset. The model uses both categorical and numerical features, with categorical features one-hot encoded and the target label binarized.

- **Algorithm:** RandomForestClassifier (scikit-learn)
- **Version:** 1.0
- **Framework:** scikit-learn 1.6+
- **Input features:** workclass, education, marital-status, occupation, relationship, race, sex, native-country, age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
- **Output:** Binary label (<=50K, >50K)

## Intended Use

This model is intended for educational and demonstration purposes in MLOps workflows. It is not intended for production or real-world deployment without further validation and fairness analysis.

## Training Data

- **Source:** UCI Census Income dataset (census_clean.csv)
- **Preprocessing:** All spaces were removed from column names and string values. Categorical features were one-hot encoded, and the label was binarized.
- **Size:** Approximately 32,000 rows.

## Evaluation Data

- 20% of the cleaned dataset was held out for evaluation as a test split.

## Metrics

The following metrics were used to evaluate model performance:
- **Precision**
- **Recall**
- **F1 Score (fbeta, beta=1)**

Performance on the overall test set:
- **F1 Score:** 0.78
- **Precision:** 0.80
- **Recall:** 0.76

Performance on data slices (example for "workclass"):
- State-gov: Precision 0.951, Recall 0.932, F1 0.941
- Self-emp-not-inc: Precision 0.953, Recall 0.890, F1 0.920
- Private: Precision 0.953, Recall 0.927, F1 0.940
- Federal-gov: Precision 0.962, Recall 0.960, F1 0.961

See `slice_output.txt` for full per-slice performance.

## Ethical Considerations

- The model may reflect biases present in the original census data.
- Sensitive attributes (e.g., race, sex, native-country) are used as features; use caution when interpreting results.
- This model is not suitable for making real-world decisions about individuals.

## Caveats and Recommendations

- Model performance may vary across different subgroups; always check slice metrics.
- Further hyperparameter tuning, feature engineering, and fairness analysis are recommended before any deployment.
- This model is for demonstration only and should not be used in production without further validation.
