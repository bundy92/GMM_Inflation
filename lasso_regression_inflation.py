# lasso_regression.py

import pandas as pd
import sklearn
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def lasso_regression(X, y):
    """
    Perform LASSO regression on the given data.

    Parameters:
    - X (pd.DataFrame): Features dataframe.
    - y (pd.Series): Target variable.

    Returns:
    - tuple: A tuple containing selected features and corresponding coefficients.
    """
    # Standardize the features
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

    # Perform LASSO regression with cross-validated regularization strength
    lasso_model = LassoCV(cv=5)
    lasso_model.fit(X_train, y_train)

    # Print selected features and corresponding coefficients
    selected_features = X.columns[lasso_model.coef_ != 0]
    lasso_coefficients = dict(zip(selected_features, lasso_model.coef_[lasso_model.coef_ != 0]))

    return selected_features, lasso_coefficients
