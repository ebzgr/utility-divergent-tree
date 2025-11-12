# Divergence Tree

Python package for identifying heterogeneous treatment effects on two outcomes (firm and consumer) using recursive partitioning.

**Requirements**: Python >= 3.9

## Installation

1. Download the package:

```bash
git clone https://github.com/ebzgr/divergence-tree  # Replace with actual repository URL
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

3. Install dependencies and package:

```bash
pip install -r requirements.txt  # Installs dependencies with specific versions
pip install -e .                 # Installs the package in editable mode
```

**Dependencies**: numpy, pandas, matplotlib, optuna, scikit-learn, econml

## Quick Start

```python
from divtree.tree import DivergenceTree

tree = DivergenceTree(max_partitions=8, min_improvement_ratio=0.01)
tree.fit(X, T, YF, YC)
region_types = tree.predict_region_type(X)  # Returns array of 1-4
```

## Methods

### DivergenceTree

Recursive partitioning algorithm that directly segments populations based on joint treatment effects.

**Algorithm**:

1. Grows maximal tree up to `max_partitions` leaves using global split selection
2. Prunes bottom-up, removing splits with `improvement_ratio` below `min_improvement_ratio`

**Split objective**:

```
g(τF, τC) = (τF - τ̄F)²/σF² + (τC - τ̄C)²/σC² + λ·φ((τF - τ̄F)(τC - τ̄C)/(σF·σC))
```

**Region types** (same for both methods):

- Region 1: τF > 0, τC > 0 (both positive)
- Region 2: τF > 0, τC ≤ 0 (firm positive, consumer negative)
- Region 3: τF ≤ 0, τC > 0 (firm negative, consumer positive)
- Region 4: τF ≤ 0, τC ≤ 0 (both negative)

**Key parameters**:

- `max_partitions`: Maximum leaves before pruning (default: 8)
- `min_improvement_ratio`: Minimum improvement to keep split (default: 0.01)
- `lambda_`: Co-movement weight (default: 1.0)
- `co_movement`: 'both', 'converge', or 'diverge' (default: 'both')

**Usage**:

```python
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna

# Tune hyperparameters
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

# Train
tree = DivergenceTree(**best_params)
tree.fit(X, T, YF, YC)

# Predict region types (1-4)
region_types = tree.predict_region_type(X)
```

### TwoStepDivergenceTree

Two-step approach: causal forests estimate treatment effects, then a classification tree predicts region types.

**Algorithm**:

1. Fit separate causal forests for YF and YC using `econml.dml.CausalForestDML`
2. Estimate τF and τC for all observations
3. Categorize into 4 region types based on effect signs (see region types above)
4. Train `sklearn.tree.DecisionTreeClassifier` to predict region types

**Usage**:

```python
from twostepdivtree.tree import TwoStepDivergenceTree

tree = TwoStepDivergenceTree(
    causal_forest_params={"n_jobs": -1, "random_state": 0},
    causal_forest_tune_params={"params": "auto"},
    classification_tree_params={"random_state": 0},
)

tree.fit(
    X, T, YF, YC,
    auto_tune_classification_tree=True,
    classification_tree_search_space={
        "max_depth": {"low": 2, "high": 15},
        "min_samples_split": {"low": 2, "high": 20},
    },
    classification_tree_tune_n_trials=30,
)

region_types = tree.predict_region_type(X)
```

## When to Use Which Method

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

## Data Format

- **X**: Feature matrix (n_samples, n_features)
- **T**: Treatment indicator (n_samples,) with values in {0, 1}
- **YF**: Firm outcome (n_samples,) - binary or continuous, may contain NaN
- **YC**: Consumer outcome (n_samples,) - binary or continuous, may contain NaN

Outcome types (binary vs continuous) are automatically detected. NaN values are handled as missing data.

## API

### DivergenceTree

- `fit(X, T, YF, YC)`: Train tree
- `predict_region_type(X)`: Predict region types (1-4)
- `leaf_effects()`: Get treatment effects for each leaf

### TwoStepDivergenceTree

- `fit(X, T, YF, YC, ...)`: Train two-step model
- `predict_region_type(X)`: Predict region types (1-4)
- `predict_treatment_effects(X)`: Predict τF and τC
- `leaf_effects()`: Get leaf effect summary
