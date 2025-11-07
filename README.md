# Divergence Tree: Auditing Optimized Interventions

A Python package for auditing marketing interventions and identifying subpopulations where firm-side and consumer-side outcomes either align (win-win) or diverge (trade-offs).

## Overview

The Divergence Tree package provides two methods for analyzing heterogeneous treatment effects on two outcomes simultaneously:

1. **DivergenceTree**: A recursive partitioning algorithm that directly segments populations based on joint treatment effects
2. **TwoStepDivergenceTree**: An alternative approach using causal forests to estimate effects, then a classification tree to predict region types

Both methods are designed to audit optimized marketing policies and identify where firm-side gains may come at the expense of consumer well-being, or where both parties benefit.

### Key Features

- **Joint Outcome Analysis**: Simultaneously analyzes firm-side (e.g., conversion) and consumer-side (e.g., satisfaction) treatment effects
- **Flexible Outcome Types**: Automatically handles binary or continuous outcomes for both firm and consumer sides
- **Missing Data Handling**: Robust handling of NaN values as general missing data indicators
- **Interpretable Segmentation**: Creates rule-based segments that are transparent and actionable
- **Region Type Classification**: Categorizes observations into 4 region types based on treatment effect signs
- **Hyperparameter Optimization**: Built-in Optuna-based tuning for optimal performance
- **Visualization**: Rich tree visualizations with color-coded segments
- **Method Comparison**: Tools for comparing different approaches side-by-side

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/divergence-tree.git
cd divergence-tree

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Dependencies

The package requires:

- `numpy>=1.24.3`
- `pandas>=2.0.3`
- `matplotlib>=3.7.2`
- `optuna>=3.2.0`
- `scikit-learn>=1.7.2`
- `econml>=0.14.0` (for TwoStepDivergenceTree)

## Quick Start

### Basic Example

The `examples/basic.py` file demonstrates a complete workflow:

```python
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna
from divtree.viz import plot_divergence_tree

# Generate or load your data
X, T, YF, YC = make_data(n=20000, seed=0, aF=0.20, aC=0.30)

# Set fixed parameters
fixed = {
    "lambda_": 1,
    "n_quantiles": 50,
    "random_state": 0,
    "co_movement": "both",
    "eps_scale": 1e-8,
}

# Define search space for tuning
search_space = {
    "max_partitions": {"low": 4, "high": 15},
    "min_improvement_ratio": {"low": 0.001, "high": 0.05, "log": True},
}

# Tune hyperparameters
best_params, best_loss = tune_with_optuna(
    X, T, YF, YC,
    fixed=fixed,
    search_space=search_space,
    n_trials=10,
    n_splits=2,
    random_state=0,
)

# Train final model
tree = DivergenceTree(**best_params)
tree.fit(X, T, YF, YC)

# Visualize
fig, ax = plot_divergence_tree(tree, figsize=(13, 7))
plt.show()
```

This example shows:

- How to generate synthetic data with binary firm outcomes and continuous consumer outcomes
- How to use hyperparameter tuning with Optuna
- How to create and visualize the divergence tree
- How to access leaf-level treatment effects

## Methods

### Method 1: DivergenceTree

The `DivergenceTree` class implements a recursive partitioning algorithm that directly identifies regions with heterogeneous treatment effects.

#### Algorithm

1. **Grow Phase**: Grows a maximal tree up to `max_partitions` leaves using global split selection
2. **Prune Phase**: Prunes the tree bottom-up, removing splits with `improvement_ratio` below `min_improvement_ratio`

#### Split Selection Objective

The algorithm uses a joint splitting criterion that considers heterogeneity in both firm-side (τF) and consumer-side (τC) treatment effects:

```
g(τF, τC) = (τF - τ̄F)²/σF² + (τC - τ̄C)²/σC² + λ·φ((τF - τ̄F)(τC - τ̄C)/(σF·σC))
```

Where:

- First two terms ensure variation in both outcomes
- Third term captures whether effects move together or apart
- φ() can be configured for convergence-seeking, divergence-seeking, or both

#### Key Parameters

- `max_partitions`: Maximum number of leaves to grow before pruning (default: 8)
- `min_improvement_ratio`: Minimum improvement ratio required to keep a split (default: 0.01)
- `lambda_`: Weight for co-movement term (default: 1.0)
- `n_quantiles`: Number of quantiles for continuous feature splits (default: 32)
- `co_movement`: Mode for co-movement term - 'both', 'converge', or 'diverge' (default: 'both')
- `random_state`: Random seed for reproducibility

#### Usage

```python
from divtree.tree import DivergenceTree

tree = DivergenceTree(
    max_partitions=10,
    min_improvement_ratio=0.01,
    lambda_=1.0,
    co_movement="both",
    random_state=0
)
tree.fit(X, T, YF, YC)

# Predict region types (1-4)
region_types = tree.predict_region_type(X)

# Get leaf-level effects
leaf_effects = tree.leaf_effects()
```

### Method 2: TwoStepDivergenceTree

The `TwoStepDivergenceTree` class implements an alternative approach using causal forests and classification trees.

#### Algorithm

1. **Step 1**: Fit separate causal forests (using `econml.dml.CausalForestDML`) for firm (YF) and consumer (YC) outcomes
2. **Step 2**: Estimate treatment effects (τF, τC) for all observations
3. **Step 3**: Categorize observations into 4 region types based on effect signs:
   - Region 1: τF > 0 and τC > 0 (both positive)
   - Region 2: τF > 0 and τC ≤ 0 (firm+, customer-)
   - Region 3: τF ≤ 0 and τC > 0 (firm-, customer+)
   - Region 4: τF ≤ 0 and τC ≤ 0 (both negative)
4. **Step 4**: Train a classification tree (using `sklearn.tree.DecisionTreeClassifier`) to predict region types from features

#### Key Features

- Automatic tuning of causal forests using `econml`'s built-in `tune()` method
- Optional automatic tuning of classification tree using Optuna
- Parallel processing support for causal forests via `n_jobs` parameter
- Region type prediction for new observations

#### Usage

```python
from twostepdivtree.tree import TwoStepDivergenceTree

tree = TwoStepDivergenceTree(
    causal_forest_params={"n_estimators": 100, "n_jobs": -1, "random_state": 0},
    causal_forest_tune_params={"params": "auto"},
    classification_tree_params={"random_state": 0},
)

tree.fit(
    X, T, YF, YC,
    auto_tune_classification_tree=True,
    classification_tree_search_space={
        "max_depth": {"low": 2, "high": 15},
        "min_samples_split": {"low": 2, "high": 20},
        "min_samples_leaf": {"low": 1, "high": 10},
    },
    classification_tree_tune_n_trials=30,
    classification_tree_tune_n_splits=2,
)

# Predict region types
region_types = tree.predict_region_type(X)

# Predict treatment effects
tauF, tauC = tree.predict_treatment_effects(X)
```

## Method Comparison

The `simulations/comparison/simulate.py` script provides a comprehensive framework for comparing both methods. The workflow is split into 4 independent steps that can be run separately:

1. **Step 1**: Generate and save training/test data
2. **Step 2**: Load data, run DivergenceTree, save results and tree
3. **Step 3**: Load data, run TwoStepDivergenceTree, save results and tree
4. **Step 4**: Load results, compare methods, and visualize trees side-by-side

### Running the Comparison

```python
# In simulate.py, comment/uncomment steps as needed:

# Step 1: Generate and save data
step1_generate_and_save_data()

# Step 2: Run DivergenceTree and save results
step2_run_divergence_tree()

# Step 3: Run TwoStepDivergenceTree and save results
step3_run_twostep_tree()

# Step 4: Compare results and visualize
step4_compare_results()
```

The comparison includes:

- Overall accuracy metrics for region type prediction
- Confusion matrices for both methods
- Per-region accuracy breakdown
- Side-by-side tree visualizations
- Saved results in pickle format for later analysis

## Region Types

Both methods categorize observations into 4 region types based on treatment effect signs:

- **Region 1** (Green): Both positive (τF > 0, τC > 0) - Win-win scenarios
- **Region 2** (Red): Firm positive, consumer negative (τF > 0, τC ≤ 0) - Trade-off: firm wins, consumer loses
- **Region 3** (Blue): Firm negative, consumer positive (τF ≤ 0, τC > 0) - Trade-off: firm loses, consumer wins
- **Region 4** (Gray): Both negative (τF ≤ 0, τC ≤ 0) - Lose-lose scenarios

## Visualization

The package provides rich visualization capabilities with color-coded segments:

- **Green**: Win-win scenarios (τF > 0, τC > 0)
- **Red**: Firm wins, consumer loses (τF > 0, τC < 0)
- **Blue**: Firm loses, consumer wins (τF < 0, τC > 0)
- **Gray**: Both lose (τF < 0, τC < 0)

Visualizations show:

- Tree structure with split rules
- Treatment effects at each node/leaf
- Color coding by region type
- Side-by-side comparisons of different methods

## Data Requirements

Both methods accept:

- **X**: Feature matrix of shape (n_samples, n_features)
- **T**: Treatment indicator of shape (n_samples,) with values in {0, 1}
- **YF**: Firm outcome of shape (n_samples,) - can be binary or continuous, may contain NaN
- **YC**: Consumer outcome of shape (n_samples,) - can be binary or continuous, may contain NaN

The algorithms automatically detect whether each outcome is binary (0/1 only) or continuous, and handle NaN values appropriately.

## Hyperparameter Tuning

Both methods support hyperparameter optimization using Optuna:

### DivergenceTree Tuning

```python
from divtree.tune import tune_with_optuna

best_params, best_loss = tune_with_optuna(
    X, T, YF, YC,
    fixed={"lambda_": 1, "n_quantiles": 50, "random_state": 0},
    search_space={
        "max_partitions": {"low": 4, "high": 15},
        "min_improvement_ratio": {"low": 0.001, "high": 0.05, "log": True},
    },
    n_trials=20,
    n_splits=5,
    random_state=0,
)
```

### TwoStepDivergenceTree Tuning

The causal forests are automatically tuned using `econml`'s built-in `tune()` method. The classification tree can be tuned automatically during `fit()`:

```python
tree.fit(
    X, T, YF, YC,
    auto_tune_classification_tree=True,
    classification_tree_search_space={...},
    classification_tree_tune_n_trials=30,
    classification_tree_tune_n_splits=2,
)
```

## Examples

### Basic Example

See `examples/basic.py` for a complete example demonstrating:

- Data generation with binary firm outcomes and continuous consumer outcomes
- Hyperparameter tuning
- Tree training and visualization
- Accessing leaf-level effects

### Method Comparison

See `simulations/comparison/simulate.py` for a comprehensive comparison framework that:

- Generates training and test datasets
- Trains both methods independently
- Compares performance metrics
- Visualizes trees side-by-side
- Saves results for reproducibility

## API Reference

### DivergenceTree

Main class: `divtree.tree.DivergenceTree`

Key methods:

- `fit(X, T, YF, YC)`: Train the tree
- `predict_region_type(X)`: Predict region types (1-4) for new observations
- `leaf_effects()`: Get treatment effects for each leaf

### TwoStepDivergenceTree

Main class: `twostepdivtree.tree.TwoStepDivergenceTree`

Key methods:

- `fit(X, T, YF, YC, ...)`: Train the two-step model
- `predict_region_type(X)`: Predict region types (1-4) for new observations
- `predict_treatment_effects(X)`: Predict treatment effects (τF, τC) for new observations
- `leaf_effects()`: Get summary of leaf effects from classification tree

## License

[Add your license information here]

## Citation

[Add citation information here]
