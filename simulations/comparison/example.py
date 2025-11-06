"""
Example: How to use the comparison data generator with divergence tree.

This shows the complete workflow:
1. Generate data with specified parameters
2. Optimize hyperparameters using cross-validation
3. Train divergence tree with best parameters
4. Visualize the tree
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from data_generator import generate_comparison_data, get_data_summary

# Add the src directory to the path to import divtree
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna
from divtree.viz import plot_divergence_tree


def run_example():
    """Run the complete example with divergence tree analysis."""

    # 1. Generate data with specified parameters
    print("Generating data...")
    X, T, YF, YC, tauF, tauC = generate_comparison_data(
        n_users=50000,
        n_features=10,
        base_subscription_prob=0.5,
        user_outcome_noise_std=0,
        random_seed=0,
        firm_effect_strength=0.1,  # Direct firm effect strength
        user_effect_strength=1.0,  # Direct user effect strength
    )

    # 2. Get data summary
    print("\nData Summary:")
    summary = get_data_summary(X, T, YF, YC)
    for key, value in summary.items():
        print(
            f"  {key}: {value:.4f}"
            if isinstance(value, (int, float))
            else f"  {key}: {value}"
        )

    # 3. Hyperparameter optimization
    print("\nOptimizing hyperparameters...")

    # Define fixed parameters (not tuned)
    fixed_params = {
        "lambda_": 1,
        "n_quantiles": 50,
        "random_state": 42,
        "co_movement": "both",
        "eps_scale": 1e-8,
    }

    # Optional: customize search space (uses defaults if not provided)
    search_space = {
        "max_partitions": {"low": 4, "high": 15},
        "min_improvement_ratio": {"low": 0.001, "high": 0.05, "log": True},
    }

    print(f"Fixed parameters: {fixed_params}")
    print(f"Search space: {search_space}")
    print("Running hyperparameter optimization...")

    # Tune hyperparameters
    best_params, best_loss = tune_with_optuna(
        X,
        T,
        YF,
        YC,
        fixed=fixed_params,
        search_space=search_space,
        n_trials=10,  # Adjust based on your needs
        n_splits=2,
        random_state=42,
    )

    print(f"\nBest parameters found: {best_params}")
    print(f"Best CV loss: {best_loss:.6f}")

    # 4. Train final tree with best parameters
    print("\nTraining final tree with best parameters...")
    tree = DivergenceTree(**best_params)
    tree.fit(X, T, YF, YC)

    # 5. Visualize the tree
    print("\nCreating tree visualization...")
    plot_divergence_tree(tree, figsize=(15, 10))
    plt.savefig("data/divergence_tree.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Tree visualization saved as 'data/divergence_tree.png'")

    print("\nExample complete!")


if __name__ == "__main__":
    run_example()
