# Importing dependencies. 
import pandas as pd
import numpy as np
import numba
import seaborn as sns
import matplotlib.pyplot as plt
from linearmodels.iv import IVGMM
from lasso_regression_inflation import lasso_regression

# Function to simulate inflation and GMM
@numba.jit(nopython=True)
def simulate_inflation_and_gmm(initial_inflation, num_years, num_simulations, true_mean_shock, true_std_dev_shock):
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

        # Generate shocks (adjust this based on your specific variables)
        shocks = np.random.normal(true_mean_shock, true_std_dev_shock, num_years)

        for i in range(num_years):
            # Update inflation with the shocks
            inflation = inflation + shocks[i]

            # Ensure inflation remains non-negative
            inflation = max(0, inflation)

        inflation_values.append(inflation)

    return np.array(inflation_values)

# GMM criterion function
@numba.jit(nopython=True)
def gmm_criterion(params, *args):
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
    criterion = ((simulated_mean - true_mean)**2 + (simulated_std_dev - true_std_dev)**2)
    return criterion

# True parameters for shocks
true_mean_shock = 0.5
true_std_dev_shock = 1.0

# Parameters for simulation
initial_inflation_rate = 2.2
num_simulation_years = 50
num_simulations = 1_000_000

# Run the Monte Carlo simulation with true parameters
true_data = simulate_inflation_and_gmm(initial_inflation_rate, num_simulation_years, num_simulations,
                                        true_mean_shock, true_std_dev_shock)

# Load your panel data (replace 'your_panel_data.csv' with your actual file)
panel_data = pd.read_csv('synthetic_g7_panel_data_enhanced.csv')

# Split data into features (X) and target (y)
X = panel_data[['Oil_Prices', 'Gas_Prices', 'GDP', 'Interest_Rates', 'Unemployment_Rate', 'Exchange_Rates', 'CPI', 'PPI', 'Labor_Market_Indicator', 'Government_Spending', 'Trade_Balance', 'Stock_Market_Index', 'Housing_Prices', 'Commodity_Prices', 'Technology_Adoption', 'Demographic_Factor']]
y = panel_data['Inflation']

# Perform LASSO regression and get selected features
selected_features, lasso_coefficients = lasso_regression(X, y)

# Specify the formula with endogenous, exogenous, and instruments
formula = 'Inflation ~ 1 + Oil_Prices + Gas_Prices  + GDP + Interest_Rates + Unemployment_Rate + Exchange_Rates + CPI + PPI + Labor_Market_Indicator + Government_Spending + Trade_Balance + Stock_Market_Index + Housing_Prices + Commodity_Prices + Technology_Adoption + Demographic_Factor'
formula_lasso = 'Inflation ~ 1 + Oil_Prices + Gas_Prices  + GDP + Interest_Rates + Unemployment_Rate + Exchange_Rates + CPI + PPI + Labor_Market_Indicator + Government_Spending + Trade_Balance + Stock_Market_Index + Housing_Prices + Commodity_Prices + Technology_Adoption + Demographic_Factor'

""" # Specify your instrument variables
# instruments = panel_data[['Oil_Prices', 'Gas_Prices', 'Current_Account_Balance', 'Other_Macro_Variables']]

# Create an IVGMM model with heteroskedasticity-robust standard errors
# iv_model = IVGMM.from_formula(formula, panel_data, instrument=instruments).fit()
iv_model = IVGMM.from_formula(formula, panel_data, weights=None).fit()

# Print the summary
print(iv_model.summary) """

# Create an IVGMM model with heteroskedasticity-robust standard errors
# iv_model = IVGMM.from_formula(formula, panel_data, instrument=instruments).fit()
iv_model = IVGMM.from_formula(formula, panel_data, weights=None).fit()

# Create an IVGMM model with the selected features
#formula_lasso = f'Inflation ~ 1 + {" + ".join(selected_features)}'
iv_model_lasso = IVGMM.from_formula(formula_lasso, panel_data, weights=None).fit()

# Print the summary of the IVGMM model with LASSO-selected features
print(iv_model_lasso.summary)
print(f'Selected features by LASSO: {selected_features}, LASSO coefficients: {lasso_coefficients}')


"""

"""

# Set Seaborn style
sns.set(style="whitegrid")

# Plot the results with Seaborn displot
plt.figure(figsize=(13, 8))

# Combined histogram and KDE for true data
sns.histplot(true_data, bins=50, kde=True, color='blue', label='True Data', edgecolor='black', linewidth=1.2)

# Combined histogram and KDE for simulated data from IVGMM
sns.histplot(simulate_inflation_and_gmm(initial_inflation_rate, num_simulation_years, num_simulations,
                                         true_mean_shock, true_std_dev_shock),
             bins=50, kde=True, color='red', label='Simulated Data (IVGMM)', edgecolor='black', linewidth=1.2)

# Combined histogram and KDE for simulated data from IVGMM with LASSO-selected features
sns.histplot(iv_model_lasso.predict(), bins=50, kde=True, color='green', label='Simulated Data (IVGMM with LASSO)', edgecolor='black', linewidth=1.2)

# Titles and labels
plt.title(f'Monte Carlo Simulation of Inflation with IVGMM (LASSO) and Shocks ({num_simulations} simulations)', fontsize=16)
plt.xlabel('Inflation Rate', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.legend(fontsize=12)

# Show the plot
plt.show()
