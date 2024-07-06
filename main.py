# main.py

# Import dependencies
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


# True parameters for shocks
true_mean_shock = 0.5
true_std_dev_shock = 1.0

# Parameters for simulation
initial_inflation_rate = 2.2
num_simulation_years = 50
num_simulations = 1_000_000

# Run the Monte Carlo simulation with true parameters
true_data = Simulation.simulate_inflation_and_gmm(initial_inflation_rate, num_simulation_years, num_simulations,
                                                   true_mean_shock, true_std_dev_shock)

# Load panel data
panel_data = pd.read_csv('synthetic_g7_panel_data_enhanced.csv')

# Split data into features (X) and target (y)
X = panel_data[['Oil_Prices', 'Gas_Prices', 'GDP', 'Interest_Rates', 'Unemployment_Rate', 'Exchange_Rates',
                'CPI', 'PPI', 'Labor_Market_Indicator', 'Government_Spending', 'Trade_Balance', 'Stock_Market_Index',
                'Housing_Prices', 'Commodity_Prices', 'Technology_Adoption', 'Demographic_Factor']]
y = panel_data['Inflation']

# Perform LASSO regression and get selected features
selected_features, lasso_coefficients = Simulation.lasso_regression(X, y)

# Specify the formula for LASSO
formula_lasso = f'Inflation ~ 1 + {" + ".join(selected_features)}'

# Create an IVGMM model with the selected features
iv_model_lasso = IVGMM.from_formula(formula_lasso, panel_data, weights=None).fit()

# Specify the formula to include the Phillips Curve relationship
formula_phillips = 'Inflation ~ 1 + Oil_Prices + Gas_Prices + GDP + Interest_Rates + Unemployment_Rate + Exchange_Rates + CPI + PPI + Labor_Market_Indicator + Government_Spending + Trade_Balance + Stock_Market_Index + Housing_Prices + Commodity_Prices + Technology_Adoption + Demographic_Factor'

# Fit the IVGMM model with the Phillips Curve relationship
iv_model_phillips = IVGMM.from_formula(formula_phillips, panel_data, weights=None).fit()

# Compare the summary of IVGMM model with Phillips Curve and IVGMM model with LASSO
print("IVGMM Model with Phillips Curve Relationship:")
print(iv_model_phillips.summary)
print("\nIVGMM Model with LASSO-Selected Features:")
print(iv_model_lasso.summary)

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Plot the results with Seaborn displot
plt.figure(figsize=(13, 8))
sns.histplot(true_data, bins=50, kde=True, color='blue', label='True Data', edgecolor='black', linewidth=1.2)
sns.histplot(Simulation.simulate_inflation_and_gmm(initial_inflation_rate, num_simulation_years, num_simulations,
                                                   true_mean_shock, true_std_dev_shock),
             bins=50, kde=True, color='red', label='Simulated Data (IVGMM)', edgecolor='black', linewidth=1.2)
sns.histplot(iv_model_lasso.predict(), bins=50, kde=True, color='green', label='Simulated Data (IVGMM with LASSO)',
             edgecolor='black', linewidth=1.2)
sns.histplot(iv_model_phillips.predict(), bins=50, kde=True, color='orange',
             label='Simulated Data (IVGMM with Phillips Curve)', edgecolor='black', linewidth=1.2)
plt.title(f'Monte Carlo Simulation of Inflation with IVGMM (LASSO) and Shocks ({num_simulations} simulations)',
          fontsize=16)
plt.xlabel('Inflation Rate', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.legend(fontsize=12)
plt.show()
