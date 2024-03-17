import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from linearmodels.iv import IVGMM
import GMM_Inflation
import synthetic_panel_data
import lasso_regression_inflation

from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to simulate inflation, energy price shocks, and additional macroeconomic variables
def simulate_inflation_and_gmm(initial_inflation, num_years, num_simulations, true_mean_shock, true_std_dev_shock):
    # ... (your existing code for simulation)
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

# Function to perform LASSO regression and return selected features
def perform_lasso_regression(X, y):
    # ... (your existing code for LASSO regression)
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

# Function to load synthetic or real panel data
def load_panel_data(is_synthetic):
    if is_synthetic:
        # ... (code to load synthetic data)
        panel_data = pd.read_csv('synthetic_g7_panel_data_enhanced.csv')

    else:
        # ... (code to load real panel data)
        pass

# Function to start the model and display results
def start_model():
    # ... (your existing code for Monte Carlo simulation, IVGMM modeling, and data visualization)
    # True parameters for shocks
    true_mean_shock = 0.5
    true_std_dev_shock = 1.0

    # Parameters for simulation
    initial_inflation_rate = 10.0
    num_simulation_years = 30
    num_simulations = 1000000

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

    """

    """

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Plot the results with Seaborn displot
    plt.figure(figsize=(13, 8))

    # Combined histogram and KDE for true data
    #sns.histplot(true_data, bins=100, kde=True, color='blue', label='True Data', edgecolor='black', linewidth=1.2)

    # Combined histogram and KDE for simulated data from IVGMM
    #sns.histplot(simulate_inflation_and_gmm(initial_inflation_rate, num_simulation_years, num_simulations,
    #                                         true_mean_shock, true_std_dev_shock),
    #             bins=100, kde=True, color='red', label='Simulated Data (IVGMM)', edgecolor='black', linewidth=1.2)

    # Combined histogram and KDE for simulated data from IVGMM with LASSO-selected features
    sns.histplot(iv_model_lasso.predict(), bins=100, kde=True, color='green', label='Simulated Data (IVGMM with LASSO)', edgecolor='black', linewidth=1.2)

    # Titles and labels
    plt.title(f'Monte Carlo Simulation of Inflation with IVGMM (LASSO) and Shocks ({num_simulations} simulations)', fontsize=16)
    plt.xlabel('Inflation Rate', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.legend(fontsize=12)
    
# Function to close the application
def close_app():
    root.destroy()

# Tkinter main window
root = tk.Tk()
root.title("G7 Inflation - Energy Analysis")

# Button to close the app
close_button = tk.Button(root, text="Close App", command=close_app)
close_button.pack(pady=10)

# Button to generate synthetic data
generate_synthetic_button = tk.Button(root, text="Generate Synthetic Data", command=lambda: load_panel_data(True))
generate_synthetic_button.pack(pady=10)

# Entry field to load synthetic or real panel data
data_entry = tk.Entry(root, width=50)
data_entry.pack(pady=10)

# Button to load panel data
load_data_button = tk.Button(root, text="Load Panel Data", command=lambda: load_panel_data(False))
load_data_button.pack(pady=10)

# Button to start the model
start_model_button = tk.Button(root, text="Start Model", command=start_model)
start_model_button.pack(pady=10)

# Label to display results
results_label = tk.Label(root, text="Results:")
results_label.pack(pady=10)

# Seaborn chart in Tkinter window
seaborn_frame = tk.Frame(root)
seaborn_frame.pack(pady=10)

# Create an interactive histogram using Seaborn
def show_seaborn_plot():
    plt.figure(figsize=(8, 6))
    
    # ... (your existing code to create Seaborn plot)

    plt.title(f'Monte Carlo Simulation of Inflation with IVGMM (LASSO) and Shocks ({num_simulations} simulations)')
    plt.xlabel('Inflation Rate')
    plt.ylabel('Frequency')

    canvas = FigureCanvasTkAgg(plt.gcf(), master=seaborn_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Button to show Seaborn plot
show_seaborn_plot_button = tk.Button(seaborn_frame, text="Show Seaborn Plot", command=show_seaborn_plot)
show_seaborn_plot_button.pack(pady=10)

# Run the Tkinter main loop
root.mainloop()
