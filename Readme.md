# Inflation Modeling with Monte Carlo Simulation and IVGMM

## Overview

This project aims to model inflation using a Monte Carlo simulation with energy price shocks and apply the Generalized Method of Moments (GMM) using the Instrumental Variables Generalized Method of Moments (IVGMM) framework.

## Project Structure

- **`data/`**: Directory containing panel data files.
- **`src/`**: Source code directory.
  - **`monte_carlo_simulation.py`**: Python module for the Monte Carlo simulation.
  - **`ivgmm_model.py`**: Python module for the IVGMM model.
  - **`visualization.py`**: Python module for result visualization.
- **`requirements.txt`**: List of project dependencies.
- **`README.md`**: Project documentation.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/inflation-modeling.git
   ```

2. Navigate to the project directory:

   ```bash
   cd inflation-modeling
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Monte Carlo Simulation

Run the Monte Carlo simulation to generate synthetic inflation data with energy price shocks.

```bash
python src/monte_carlo_simulation.py
```

### 2. IVGMM Modeling

Apply the IVGMM model to the panel data for inflation modeling.

```bash
python src/ivgmm_model.py
```

### 3. Result Visualization

Visualize the results of the Monte Carlo simulation and IVGMM modeling.

```bash
python src/visualization.py
```

## Configuration

Adjust simulation and modeling parameters in the respective Python modules (`monte_carlo_simulation.py` and `ivgmm_model.py`).

## Data

Panel data for G7 nations spanning three decades is required for IVGMM modeling. Ensure the data is formatted correctly and located in the `data/` directory.

## Contributing

If you would like to contribute to this project, please follow the [Contributing Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Special thanks to [Author Name] for their valuable contributions.

## Contact

For inquiries and support, please contact [Your Name] at [your.email@example.com].
