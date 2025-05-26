# TASK-1

This project performs a complete preprocessing pipeline on the Titanic dataset using Python and popular data science libraries. Key steps include:

1. Initial Data Exploration: Displayed sample rows, data types, summary statistics, and missing values.

2. Missing Value Handling:

* Filled missing Age with the mean.
* Filled missing Embarked with the mode.
* Extracted Deck from Cabin and imputed missing Deck values based on Pclass.
* Dropped the original Cabin column.

3. Outlier Removal: Removed outliers in Age and Fare using the IQR method.

4. Feature Encoding:

* Label encoded Sex.
* One-hot encoded Embarked and Deck (excluding first category to prevent multicollinearity).

5. Feature Scaling: Normalized Age and Fare using Min-Max Scaling.

6. Visualization: Used boxplots to visualize Age and Fare distributions before and after outlier removal.

This cleaned dataset is ready for machine learning model training.
