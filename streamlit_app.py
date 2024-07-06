import streamlit as st
import pandas as pd
import numpy as np
import numba
import seaborn as sns
import matplotlib.pyplot as plt
from linearmodels.iv import IVGMM
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

# Disable warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define a class for simulation
class Simulation:
    def __init__(self):
        pass

    @staticmethod
    @numba.jit(nopython=True)
    def simulate_inflation_and_gmm(initial_inflation: float, num_years: int, num_simulations: int,
                                   true_mean_shock: float, true_std_dev_shock: float) -> np.ndarray:
        """
        Simulate inflation over multiple years with Gaussian shocks.

        Parameters:
        - initial_inflation: Initial inflation rate.
        - num_years: Number of simulation years.
        - num_simulations: Number of simulation runs.
        - true_mean_shock: Mean of the Gaussian shock.
        - true_std_dev_shock: Standard deviation of the Gaussian shock.

        Returns:
        - Array of simulated inflation values.
        """
        inflation_values = []

        for _ in range(num_simulations):
            inflation = initial_inflation

            # Generate shocks
            shocks = np.random.normal(true_mean_shock, true_std_dev_shock, num_years)

            for i in range(num_years):
                inflation = max(0, inflation + shocks[i])  # Ensure inflation remains non-negative

            inflation_values.append(inflation)

        return np.array(inflation_values)

    @staticmethod
    @numba.jit(nopython=True)
    def gmm_criterion(params: Tuple[float, float], *args) -> float:
        """
        GMM criterion function for optimization.

        Parameters:
        - params: Parameters to optimize (true_mean, true_std_dev).
        - args: Arguments containing simulated and true data.

        Returns:
        - GMM criterion value.
        """
        simulated_data, true_data = args
        simulated_mean = np.mean(simulated_data)
        simulated_std_dev = np.std(simulated_data)

        true_mean, true_std_dev = params
        criterion = ((simulated_mean - true_mean) ** 2 + (simulated_std_dev - true_std_dev) ** 2)
        return criterion

    @staticmethod
    def lasso_regression(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Perform LASSO regression on the given data.

        Parameters:
        - X: Features dataframe.
        - y: Target variable.

        Returns:
        - Tuple containing selected features and corresponding coefficients.
        """
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

        lasso_model = LassoCV(cv=5)
        lasso_model.fit(X_train, y_train)

        selected_features = X.columns[lasso_model.coef_ != 0]
        lasso_coefficients = dict(zip(selected_features, lasso_model.coef_[lasso_model.coef_ != 0]))

        return selected_features, lasso_coefficients


def main():
    # Page title
    st.title("Inflation Simulation and Modeling")

    # Sidebar with parameters
    st.sidebar.header("Simulation Parameters")
    initial_inflation_rate = st.sidebar.slider("Initial Inflation Rate", min_value=0.0, max_value=10.0, value=2.2)
    true_mean_shock = st.sidebar.slider("True Mean Shock", min_value=0.0, max_value=2.0, value=0.5)
    true_std_dev_shock = st.sidebar.slider("True Std Dev Shock", min_value=0.0, max_value=5.0, value=1.0)
    num_simulation_years = st.sidebar.slider("Number of Simulation Years", min_value=1, max_value=100, value=50)
    num_simulations = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=1000000, step=1000, value=100000)

    # Run the simulation
    true_data = Simulation.simulate_inflation_and_gmm(initial_inflation_rate, num_simulation_years, num_simulations,
                                                       true_mean_shock, true_std_dev_shock)

    # Display true data
    st.subheader("True Inflation Data")
    st.write(true_data)

    # Plot histogram of true data
    st.subheader("Histogram of True Inflation Data")
    plt.figure(figsize=(10, 6))
    sns.histplot(true_data, bins=50, kde=True)
    st.pyplot()

    # Data loading and preprocessing
    panel_data = pd.read_csv('synthetic_g7_panel_data_enhanced.csv')
    X = panel_data[['Oil_Prices', 'Gas_Prices', 'GDP', 'Interest_Rates', 'Unemployment_Rate', 'Exchange_Rates',
                    'CPI', 'PPI', 'Labor_Market_Indicator', 'Government_Spending', 'Trade_Balance', 'Stock_Market_Index',
                    'Housing_Prices', 'Commodity_Prices', 'Technology_Adoption', 'Demographic_Factor']]
    y = panel_data['Inflation']

    # Perform LASSO regression
    selected_features, lasso_coefficients = Simulation.lasso_regression(X, y)
    st.subheader("LASSO Regression Results")
    st.write("Selected Features:", selected_features)
    st.write("LASSO Coefficients:", lasso_coefficients)

    # IVGMM Model with LASSO-Selected Features
    formula_lasso = f'Inflation ~ 1 + {" + ".join(selected_features)}'
    iv_model_lasso = IVGMM.from_formula(formula_lasso, panel_data, weights=None).fit()

    # Display IVGMM summary with LASSO
    st.subheader("IVGMM Model with LASSO-Selected Features")
    st.write(iv_model_lasso.summary)

    # IVGMM Model with Phillips Curve Relationship
    formula_phillips = 'Inflation ~ 1 + Oil_Prices + Gas_Prices + GDP + Interest_Rates + Unemployment_Rate + Exchange_Rates + CPI + PPI + Labor_Market_Indicator + Government_Spending + Trade_Balance + Stock_Market_Index + Housing_Prices + Commodity_Prices + Technology_Adoption + Demographic_Factor'
    iv_model_phillips = IVGMM.from_formula(formula_phillips, panel_data, weights=None).fit()

    # Display IVGMM summary with Phillips Curve
    st.subheader("IVGMM Model with Phillips Curve Relationship")
    st.write(iv_model_phillips.summary)

    # Plot histogram of simulated data
    st.subheader("Histogram of Simulated Inflation Data")
    plt.figure(figsize=(10, 6))
    sns.histplot(iv_model_lasso.predict(), bins=50, kde=True, color='green', label='Simulated Data (IVGMM with LASSO)')
    sns.histplot(iv_model_phillips.predict(), bins=50, kde=True, color='orange', label='Simulated Data (IVGMM with Phillips Curve)')
    st.pyplot()

if __name__ == "__main__":
    main()
