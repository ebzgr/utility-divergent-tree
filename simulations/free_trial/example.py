"""
Example: How to use configurations to generate data and run analysis.

This shows the basic workflow:
1. Create a configuration
2. Generate data with that config
3. Run analysis with that config
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from synthetic_data_generator import generate_data
from helper import (
    plot_latent_overview,
    print_summary,
    make_group_table_from_arrays,
    plot_six_panels_outcomes_and_pi,
)
from config import DataGenerationConfig, AnalysisConfig, FreeTrialConfig
from divtree.tree import DivergenceTree
from divtree.tune import tune_with_optuna_partial
from divtree.viz import plot_divergence_tree

# ===================== GLOBAL PATHS =====================
DATA_DIR = "data"
RESULTS_DIR = "results"
DATA_SAVE_ADDR = f"{DATA_DIR}/example_data.pickle"
# ======================================================


def run_example():
    """Run the complete example with custom configuration."""

    # 1. Create a custom configuration - Check the full list of confits in config.py
    config = FreeTrialConfig(
        data_generation=DataGenerationConfig(
            N_USERS=100000,  # Smaller dataset for faster testing
            N_FEATURES=50,
            NOISE_SCALE=1.0,
            USEFULNESS_BIPOLARITY=2,
            NOVELTY_BIPOLARITY=2,
            SUNK_COST_BIPOLARITY=2,
        ),
        analysis=AnalysisConfig(
            N_TUNING_TRIALS=20,
            LAMBDA_=2,
        ),
    )

    # 2. Generate data with the config
    print("Generating data...")
    params = {
        "n": config.data_generation.N_USERS,
        "p": config.data_generation.N_FEATURES,
        "seed": config.data_generation.RANDOM_SEED,
        "gate_indices": config.data_generation.GATE_INDICES,
        "gate_thresholds": config.data_generation.GATE_THRESHOLDS,
        "v": config.data_generation.NOISE_SCALE,
        "m_U": config.data_generation.USEFULNESS_BIPOLARITY,
        "m_N": config.data_generation.NOVELTY_BIPOLARITY,
        "m_S": config.data_generation.SUNK_COST_BIPOLARITY,
        "a_u": config.data_generation.USEFULNESS_SHORT_WEIGHT,
        "a_s": config.data_generation.SUNK_COST_LONG_WEIGHT,
        "a0": config.data_generation.BASELINE_SUBSCRIPTION,
        "a_p": config.data_generation.VALUE_SUBSCRIPTION_WEIGHT,
    }

    X, L, T, YF, YC, P_short, P_long, pi = generate_data(**params)

    # Descriptive analysis of the generated data
    print("Data Summary")
    print_summary(YF, T, L)
    table = make_group_table_from_arrays(T, YF, YC, L["U"], L["N"], L["S"])
    print(table)
    fig, axes = plot_six_panels_outcomes_and_pi(T, YF, YC, pi, L, nbins=20)
    # Create comprehensive DataFrame with all data
    df = pd.DataFrame(
        {
            # Features (first 10 columns for brevity)
            **{f"X_{i}": X[:, i] for i in range(min(10, X.shape[1]))},
            # Latent variables
            "U": L["U"],
            "N": L["N"],
            "S": L["S"],
            # Treatment and outcomes
            "Treat": T,
            "YF": YF,
            "YC": YC,
            # Perceived usefulness scores
            "P_short": P_short,
            "P_long": P_long,
            "P": np.where(T == 0, P_short, P_long),  # Combined P based on treatment
            # Subscription probability
            "pi": pi,
        }
    )

    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save DataFrame as pickle
    df.to_pickle(DATA_SAVE_ADDR)
    print(f"Data saved to: {DATA_SAVE_ADDR}")

    # 3. Run analysis with the config
    print("Running analysis...")
    analysis_config = config.analysis

    # Fixed parameters
    fixed = {
        "honest": analysis_config.HONEST,
        "min_leaf_treated": analysis_config.MIN_LEAF_TREATED,
        "min_leaf_control": analysis_config.MIN_LEAF_CONTROL,
        "min_leaf_conv_treated": analysis_config.MIN_LEAF_CONV_TREATED,
        "min_leaf_conv_control": analysis_config.MIN_LEAF_CONV_CONTROL,
        "random_state": analysis_config.RANDOM_STATE,
        "co_movement": analysis_config.CO_MOVEMENT,
        "lambda_": analysis_config.LAMBDA_,
    }

    # Search space
    search_space = {
        "max_depth": {
            "type": "int",
            "low": 0,
            "high": 6,
        },
        "min_leaf_n": {
            "type": "int",
            "low": int(0.01 * X.shape[0]),
            "high": int(0.1 * X.shape[0]),
            "step": int(0.01 * X.shape[0]),
        },
        "n_quantiles": {
            "type": "int",
            "low": 32,
            "high": 128,
            "step": 32,
        },
    }

    # Tune hyperparameters
    best_params, best_score = tune_with_optuna_partial(
        X,
        T,
        YF,
        YC,
        fixed=fixed,
        search_space=search_space,
        valid_fraction=analysis_config.VALID_FRACTION,
        n_trials=analysis_config.N_TUNING_TRIALS,
        random_state=analysis_config.RANDOM_STATE,
    )

    # Build final tree
    tree = DivergenceTree(**best_params)
    tree.fit(X, T, YF, YC)

    print("Example complete!")
    print(f"Best score: {best_score:.4f}")
    print(f"Best params: {best_params}")

    # Show the resulting tree
    print("\n" + "=" * 60)
    print("RESULTING DIVERGENCE TREE")
    print("=" * 60)

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Plot and save the tree
    fig, ax = plot_divergence_tree(
        tree,
        figsize=(15, 8),
        h_spacing=3.5,
        v_spacing=2.5,
        node_width=3.0,
        node_height=1.2,
    )
    plt.savefig(f"{RESULTS_DIR}/example_tree.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Tree visualization saved to: {RESULTS_DIR}/example_tree.png")


if __name__ == "__main__":
    run_example()
