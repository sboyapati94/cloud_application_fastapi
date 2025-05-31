# Model Card

## Model Details
This model is a RandomForestClassifier trained to predict whether a person's income exceeds $50K/year based on the UCI Census Income dataset. The model uses both categorical and numerical features, with categorical features one-hot encoded and the target label binarized.

- **Algorithm:** RandomForestClassifier (scikit-learn)
- **Version:** 1.0
- **Framework:** scikit-learn 1.6+
- **Input features:** workclass, education, marital-status, occupation, relationship, race, sex, native-country, age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
- **Output:** Binary label (<=50K, >50K)

## Intended Use
This model is intended for educational and demonstration purposes in MLOps workflows. It is not intended for production or real-world deployment without further validation and fairness analysis.

## Training Data
- **Source:** UCI Census Income dataset (census_clean.csv)
- **Preprocessing:** All spaces removed from column names and string values; categorical features one-hot encoded; label binarized.
- **Size:** ~32,000 rows

## Evaluation Data
- 20% of the cleaned dataset was held out for evaluation (test split).

## Metrics
- **Precision, Recall, F1 (fbeta, beta=1):** Computed on the test set.
- **Slice metrics:** Model performance is also evaluated on slices of the data for each categorical feature.

_Example (fill in with your actual results):_
- Overall test set F1: 0.78
- Overall test set Precision: 0.80
- Overall test set Recall: 0.76
- See slice_metrics.json for per-slice performance.

## Ethical Considerations
- The model may reflect biases present in the original census data.
- Sensitive attributes (e.g., race, sex, native-country) are used as features; use caution when interpreting results.
- Not suitable for making real-world decisions about individuals.

## Caveats and Recommendations
- Model performance may vary across different subgroups; always check slice metrics.
- Further hyperparameter tuning, feature engineering, and fairness analysis are recommended before any deployment.
- This model is for demonstration only and should not be used in production without further validation.
