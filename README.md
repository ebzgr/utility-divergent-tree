# Divergence Tree

Python package for identifying heterogeneous treatment effects on two outcomes using recursive partitioning to understand relationships and trade-offs.

**Requirements**: Python >= 3.9

## What is Divergence Tree?

Divergence Tree is a machine learning algorithm designed to identify heterogeneous treatment effects when you have **two outcomes of interest** and want to understand how treatment effects differ between them. The algorithm segments populations into regions where treatment effects on the two outcomes diverge or converge, revealing trade-offs and relationships.

### Purpose

In many contexts, treatments (e.g., interventions, policies, features) can have different effects on two outcomes of interest. Understanding these relationships and trade-offs is crucial for decision-making. Common applications include:

- **Firm vs Consumer outcomes**: A price increase might boost firm revenue but reduce consumer satisfaction
- **Long-term vs Short-term outcomes**: A marketing campaign might increase immediate sales but reduce long-term brand loyalty
- **Efficiency vs Quality outcomes**: Process changes might improve efficiency but reduce quality
- **Any two competing or complementary outcomes**: Where you need to understand how treatment effects vary across both dimensions

Divergence Tree segments your population into regions where treatment effects on the two outcomes diverge or converge, helping you identify:

- **Win-win regions**: Where treatments benefit both outcomes
- **Trade-off regions**: Where treatments help one outcome but harm the other
- **Lose-lose regions**: Where treatments harm both outcomes

### Algorithm Overview

The package provides two methods:

1. **DivergenceTree**: Directly optimizes a split objective function that measures divergence between treatment effects on the two outcomes. It grows a tree by recursively partitioning the feature space, then prunes splits that don't improve the objective.

2. **TwoStepDivergenceTree**: A two-step approach that first estimates treatment effects using causal forests, then trains a classification tree to predict region types based on those estimates.

Both methods categorize observations into 4 region types based on the signs of treatment effects:

- **Region 1**: τ₁ > 0, τ₂ > 0 (both positive - win-win)
- **Region 2**: τ₁ > 0, τ₂ ≤ 0 (first outcome positive, second negative - trade-off favoring first)
- **Region 3**: τ₁ ≤ 0, τ₂ > 0 (first outcome negative, second positive - trade-off favoring second)
- **Region 4**: τ₁ ≤ 0, τ₂ ≤ 0 (both negative - lose-lose)

Where τ₁ is the treatment effect on the first outcome and τ₂ is the treatment effect on the second outcome. In the firm/consumer example, these would be firm effects (τF) and consumer effects (τC).

## Installation

1. Download the package:

```bash
git clone https://github.com/ebzgr/divergence-tree
cd divergence-tree
```

2. Create a virtual environment:

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install the package (dependencies are installed automatically):

```bash
pip install -e .
```

**Dependencies**: numpy, pandas, matplotlib, optuna, scikit-learn, econml

## Usage

### Data Format

Before using the algorithms, prepare your data:

- **X**: Feature matrix of shape `(n_samples, n_features)` - characteristics of each observation (e.g., user demographics, product features, time periods)
- **T**: Treatment indicator of shape `(n_samples,)` with values in {0, 1} - whether each observation received the treatment
- **YF**: First outcome of shape `(n_samples,)` - binary or continuous, may contain NaN (e.g., firm revenue, short-term sales, efficiency metrics)
- **YC**: Second outcome of shape `(n_samples,)` - binary or continuous, may contain NaN (e.g., consumer satisfaction, long-term loyalty, quality metrics)

**Note**: YF and YC can represent any two outcomes of interest. The naming (F/C) is a convention from the firm/consumer example, but you can use any two outcomes.

Outcome types (binary vs continuous) are automatically detected. NaN values are handled as missing data.

### Using DivergenceTree

```python
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna

# Optional: Tune hyperparameters using cross-validation
best_params, best_loss = tune_with_optuna(
    X, T, YF, YC,
    fixed={"lambda_": 1, "random_state": 0},
    search_space={
        "max_partitions": {"low": 4, "high": 15},
        "min_improvement_ratio": {"low": 0.001, "high": 0.05, "log": True},
    },
    n_trials=20,
    n_splits=5,
)

# Train the tree
tree = DivergenceTree(**best_params)
tree.fit(X, T, YF, YC)

# Predict region types (returns array of 1-4)
region_types = tree.predict_region_type(X)

# Get treatment effects for each leaf
leaf_effects = tree.leaf_effects()
```

**Key parameters**:

- `max_partitions`: Maximum leaves before pruning (default: 8)
- `min_improvement_ratio`: Minimum improvement to keep split (default: 0.01)
- `lambda_`: Co-movement weight in split objective (default: 1.0)
- `co_movement`: 'both', 'converge', or 'diverge' (default: 'both')

### Using TwoStepDivergenceTree

```python
from twostepdivtree.tree import TwoStepDivergenceTree

# Initialize the two-step model
tree = TwoStepDivergenceTree(
    causal_forest_params={"n_jobs": -1, "random_state": 0},
    causal_forest_tune_params={"params": "auto"},
    classification_tree_params={"random_state": 0},
)

# Train with optional classification tree tuning
tree.fit(
    X, T, YF, YC,
    auto_tune_classification_tree=True,
    classification_tree_search_space={
        "max_depth": {"low": 2, "high": 15},
        "min_samples_split": {"low": 2, "high": 20},
    },
    classification_tree_tune_n_trials=30,
)

# Predict region types
region_types = tree.predict_region_type(X)

# Predict treatment effects (optional)
tauF, tauC = tree.predict_treatment_effects(X)
```

### When to Use Which Method

- **DivergenceTree**: Direct optimization of joint treatment effects. Better when you want a single unified model and have sufficient data for tuning.
- **TwoStepDivergenceTree**: Separates effect estimation from classification. Better when you need interpretable treatment effect estimates or want to leverage causal forest's robustness.

## Examples

### Basic Example

`examples/basic.py`: Complete workflow with data generation, hyperparameter tuning, and visualization.

### Method Comparison

`simulations/comparison/simulate.py`: Compare both methods on the same dataset. Four independent steps:

1. Generate and save data
2. Run DivergenceTree, save results
3. Run TwoStepDivergenceTree, save results
4. Compare results and visualize

## API Reference

### DivergenceTree

- `fit(X, T, YF, YC)`: Train the tree on data
- `predict_region_type(X)`: Predict region types (1-4) for new observations
- `leaf_effects()`: Get treatment effects for each leaf node

### TwoStepDivergenceTree

- `fit(X, T, YF, YC, ...)`: Train the two-step model
- `predict_region_type(X)`: Predict region types (1-4) for new observations
- `predict_treatment_effects(X)`: Predict τF and τC for new observations
- `leaf_effects()`: Get leaf effect summary
