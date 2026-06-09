Key Observations

- Dataset is heavily imbalanced (~X% fraud).
- Fraud transactions tend to have [higher/lower] purchase values.
- Certain browsers/sources show elevated fraud rates.

Handling Class Imbalance with SMOTE

Justification: The dataset is severely imbalanced (<10% fraud). SMOTE is preferred over
random oversampling because it synthesizes new minority samples rather than duplicating
existing ones, reducing overfitting. We apply SMOTE only on the training set to avoid
data leakage into the test set.

feature engineering doc
Feature Engineering & Preprocessing

Covers:

1. Geolocation merge (IP → Country)
2. Temporal features
3. Transaction velocity
4. Encoding & scaling
5. Class imbalance handling (SMOTE)
