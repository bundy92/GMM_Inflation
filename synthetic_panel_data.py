# Import necessary libraries
import pandas as pd
import numpy as np
import numba

# Function to simulate inflation, energy price shocks, and additional macroeconomic variables
@numba.jit(nopython=True)
def simulate_synthetic_data(g7_countries, start_year, end_year, true_mean_shock, true_std_dev_shock):
    """
    Simulates synthetic panel data for G7 countries based on specified parameters.

    Parameters:
    - g7_countries (list): List of G7 countries.
    - start_year (int): Starting year for simulation.
    - end_year (int): Ending year for simulation.
    - true_mean_shock (float): True mean shock for inflation.
    - true_std_dev_shock (float): True standard deviation of shocks.

    Returns:
    - pd.DataFrame: Simulated synthetic panel data.
    """
    np.random.seed(42)  # Set seed for reproducibility

    # Generate synthetic panel data
    data = pd.DataFrame({
        'Year': np.repeat(np.arange(start_year, end_year + 1), len(g7_countries)),
        'Country': np.tile(g7_countries, end_year - start_year + 1),
        'Inflation': np.nan,
        'Oil_Prices': np.nan,
        'Gas_Prices': np.nan,
        'GDP': np.nan,
        'Interest_Rates': np.nan,
        'Unemployment_Rate': np.nan,
        'Exchange_Rates': np.nan,
        'CPI': np.nan,
        'PPI': np.nan,
        'Labor_Market_Indicator': np.nan,
        'Government_Spending': np.nan,
        'Trade_Balance': np.nan,
        'Stock_Market_Index': np.nan,
        'Housing_Prices': np.nan,
        'Commodity_Prices': np.nan,
        'Technology_Adoption': np.nan,
        'Demographic_Factor': np.nan
    })

    # Simulate data for each variable
    for country in g7_countries:
        for year in range(start_year, end_year + 1):
            idx = (data['Country'] == country) & (data['Year'] == year)

            # Simulate Inflation
            inflation = np.random.normal(true_mean_shock, true_std_dev_shock)
            data.loc[idx, 'Inflation'] = max(0, inflation)

            # Simulate Energy Price Shocks
            data.loc[idx, 'Oil_Prices'] = np.random.normal(50, 10)  # Adjust as needed
            data.loc[idx, 'Gas_Prices'] = np.random.normal(5, 2)  # Adjust as needed

            # Simulate additional macroeconomic variables
            data.loc[idx, 'GDP'] = np.random.normal(1000, 200)  # Adjust as needed
            data.loc[idx, 'Interest_Rates'] = np.random.uniform(1, 5)  # Adjust as needed
            data.loc[idx, 'Unemployment_Rate'] = np.random.uniform(3, 10)  # Adjust as needed
            data.loc[idx, 'Exchange_Rates'] = np.random.normal(1, 0.1)  # Adjust as needed
            data.loc[idx, 'CPI'] = np.random.normal(1000, 200)  # Adjust as needed
            data.loc[idx, 'PPI'] = np.random.uniform(1, 5)  # Adjust as needed
            data.loc[idx, 'Labor_Market_Indicator'] = np.random.uniform(3, 10)  # Adjust as needed
            data.loc[idx, 'Government_Spending'] = np.random.normal(1, 0.1)  # Adjust as needed
            data.loc[idx, 'Trade_Balance'] = np.random.normal(1, 0.1)  # Adjust as needed
            data.loc[idx, 'Stock_Market_Index'] = np.random.normal(1, 0.1)  # Adjust as needed
            data.loc[idx, 'Housing_Prices'] = np.random.normal(1, 0.1)  # Adjust as needed
            data.loc[idx, 'Commodity_Prices'] = np.random.normal(1, 0.1)  # Adjust as needed
            data.loc[idx, 'Technology_Adoption'] = np.random.normal(1, 0.1)  # Adjust as needed
            data.loc[idx, 'Demographic_Factor'] = np.random.normal(1, 0.1)  # Adjust as needed
            # ... add more variables and simulation logic for each

    return data

# Set parameters
g7_countries = ['Canada', 'France', 'Germany', 'Italy', 'Japan', 'United Kingdom', 'United States']
start_year = 1970
end_year = 2020
true_mean_shock = 0.5  # True mean shock for inflation
true_std_dev_shock = 1.0  # True standard deviation of shocks

# Generate synthetic data
synthetic_data = simulate_synthetic_data(g7_countries, start_year, end_year, true_mean_shock, true_std_dev_shock)

# Save synthetic data to CSV
synthetic_data.to_csv('synthetic_g7_panel_data_enhanced.csv', index=False)
