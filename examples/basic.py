# examples/simplest_simulate.py
import numpy as np
import matplotlib.pyplot as plt
from divtree import DivergenceTree
from divtree.viz import plot_divergence_tree

import pdb


def make_data(n=4000, seed=0, aF=0.1, aC=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))  # we only really use X[:,0] and X[:,1]
    T = rng.integers(0, 2, size=n)

    # Your rules:
    # τF(x) > 0 iff X0 > 0; else < 0
    tauF_true = np.where(X[:, 0] > 0, +aF, -aF)

    # τC(x) > 0 iff X1 > 0; else < 0
    tauC_true = np.where(X[:, 1] > -1, +aC, -aC)

    # Base conversion probability
    p0 = np.clip(0.25 + 0.05 * np.tanh(X[:, 0]), 0.05, 0.95)
    p1 = np.clip(p0 + tauF_true, 0.01, 0.99)

    # Realized conversion YF (binary)
    YF0 = rng.binomial(1, p0)
    YF1 = rng.binomial(1, p1)
    YF = np.where(T == 1, YF1, YF0).astype(int)

    # Consumer outcome (observed only for converters)
    base_C = 0.5 + 0.2 * np.tanh(X[:, 1])
    noise = rng.normal(0, 0.35, size=n)
    YC_treat = base_C + tauC_true + noise
    YC_ctrl = base_C + noise

    YC = np.full(n, np.nan)
    mask_treat_conv = (T == 1) & (YF == 1)
    mask_ctrl_conv = (T == 0) & (YF == 1)
    YC[mask_treat_conv] = YC_treat[mask_treat_conv]
    YC[mask_ctrl_conv] = YC_ctrl[mask_ctrl_conv]

    return X, T, YF, YC


if __name__ == "__main__":
    # pip install optuna  (first time)
    from divtree.tune import tune_with_optuna_partial

    # --- simulate data (or your real data) ---
    X, T, YF, YC = make_data(n=200000, seed=0, aF=0.20, aC=0.30)

    # --- freeze what you want fixed ---
    fixed = {
        "honest": True,
        "lambda_": 1,
        "n_quantiles": 50,
        "min_leaf_treated": 1,
        "min_leaf_control": 1,
        "min_leaf_conv_treated": 1,
        "min_leaf_conv_control": 1,
        "random_state": 0,  # for reproducibility
    }

    # --- tune only these two ---
    search_space = {
        "max_depth": {"type": "int", "low": 2, "high": 6},
        "min_leaf_n": {"type": "int", "low": 500, "high": 4000, "step": 250},
    }

    best_params, best_score = tune_with_optuna_partial(
        X,
        T,
        YF,
        YC,
        fixed=fixed,
        search_space=search_space,
        valid_fraction=0.2,
        n_trials=10,
        random_state=123,
    )
    print("Best params:", best_params)
    print("Best validation score:", best_score)

    # --- fit final model on full data with best params ---
    tree = DivergenceTree(**best_params).fit(X, T, YF, YC)
    print(f"Root: tauF={tree.root_.tauF:.4f}, tauC={tree.root_.tauC:.4f}")

    # --- plot ---
    fig, ax = plot_divergence_tree(
        tree,
        figsize=(13, 7),
        h_spacing=3.2,
        v_spacing=2.4,
        node_width=2.8,
        node_height=1.1,
    )
    plt.show()
