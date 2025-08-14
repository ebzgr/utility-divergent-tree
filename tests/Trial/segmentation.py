import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from divtree import DivergenceTree
from divtree.viz import plot_divergence_tree
from divtree.tune import tune_with_optuna_partial

pickle_path = "tests/Trial/data/sim2.pickle"


def load_and_prepare_data():
    """
    Load the pickle data and prepare it for the divergence tree algorithm.
    Returns X, T, YF, YC arrays.
    """
    # Load the data
    df = pd.read_pickle(pickle_path)

    # Extract features (X columns)
    X_cols = [col for col in df.columns if col.startswith("X_")]
    X = df[X_cols].values

    # Extract treatment and outcomes
    T = df["Treat"].values
    YF = df["YF"].values
    YC = df["YC"].values

    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Treatment distribution: {np.bincount(T)}")
    print(f"Subscription rate: {YF.mean():.3f}")
    print(f"Consumer outcome available for {np.sum(~np.isnan(YC))} samples")

    return X, T, YF, YC


def run_segmentation_analysis(X, T, YF, YC, n_trials=20):
    """
    Run the divergence tree segmentation analysis with hyperparameter tuning.
    """
    print("\n" + "=" * 50)
    print("RUNNING DIVERGENCE TREE SEGMENTATION")
    print("=" * 50)

    # --- freeze what you want fixed ---
    fixed = {
        "honest": True,
        "min_leaf_treated": 1,
        "min_leaf_control": 1,
        "min_leaf_conv_treated": 1,
        "min_leaf_conv_control": 1,
        "random_state": 0,  # for reproducibility
        "lambda_": 3,
    }

    # --- tune these hyperparameters ---
    # example search_space
    search_space = {
        "max_depth": {"type": "int", "low": 2, "high": 6},
        "min_leaf_n": {
            "type": "int",
            "low": int(0.01 * X.shape[0]),
            "high": int(0.05 * X.shape[0]),
            "step": 100,
        },
        "n_quantiles": {"type": "int", "low": 32, "high": 128, "step": 32},  # optional
    }

    print("Tuning hyperparameters...")
    best_params, best_score = tune_with_optuna_partial(
        X,
        T,
        YF,
        YC,
        fixed=fixed,
        search_space=search_space,
        valid_fraction=0.2,
        n_trials=n_trials,
        random_state=0,
    )

    print(f"Best params: {best_params}")

    # --- fit final model on full data with best params ---
    print("\nFitting final model...")
    tree = DivergenceTree(**best_params).fit(X, T, YF, YC)

    return tree, best_params, best_score


def get_leaf_nodes(tree, X):
    """
    Get leaf node objects for each sample using the tree's predict_leaf method.
    Returns array of leaf node objects.
    """
    return tree.predict_leaf(X)


def analyze_segments(tree, X, T, YF, YC, df=None):
    """
    Analyze the segments found by the tree.
    """
    print("\n" + "=" * 50)
    print("SEGMENT ANALYSIS")
    print("=" * 50)

    # Get leaf nodes directly
    leaf_nodes = get_leaf_nodes(tree, X)

    # Get unique leaf nodes by converting to IDs first
    leaf_ids = np.array([id(node) for node in leaf_nodes])
    unique_leaf_ids = np.unique(leaf_ids)

    # Convert back to unique leaf nodes
    unique_leaf_nodes = []
    for leaf_id in unique_leaf_ids:
        # Find the first occurrence of this leaf_id
        idx = np.where(leaf_ids == leaf_id)[0][0]
        unique_leaf_nodes.append(leaf_nodes[idx])

    print(f"Found {len(unique_leaf_nodes)} segments")

    # Load the original data to get latent variables
    if df is None:
        df = pd.read_pickle(pickle_path)

    # Extract latent variables
    U = df["U"].values
    N = df["N"].values
    S = df["S"].values

    # Create summary table
    segment_summary = []

    print(
        f"\n{'Segment':<8} {'Size':<8} {'tauF':<10} {'tauC':<10} {'U_avg':<10} {'N_avg':<10} {'S_avg':<10}"
    )
    print("-" * 80)

    # Analyze each segment
    for leaf_node in unique_leaf_nodes:
        # Create mask by comparing node IDs
        mask = np.array([id(node) == id(leaf_node) for node in leaf_nodes])
        n_samples = np.sum(mask)

        # Get treatment effects directly from the leaf node
        tauF = leaf_node.tauF
        tauC = leaf_node.tauC

        # Calculate average latent variables for this segment
        U_avg = U[mask].mean()
        N_avg = N[mask].mean()
        S_avg = S[mask].mean()

        # Calculate segment statistics
        treat_mask = T[mask] == 1
        control_mask = T[mask] == 0

        if np.sum(treat_mask) > 0 and np.sum(control_mask) > 0:
            conv_treat = YF[mask][treat_mask].mean()
            conv_control = YF[mask][control_mask].mean()

            # Consumer outcome (only for subscribers)
            yc_treat = YC[mask][treat_mask & (YF[mask] == 1)]
            yc_control = YC[mask][control_mask & (YF[mask] == 1)]

            yc_treat_mean = np.nanmean(yc_treat) if len(yc_treat) > 0 else np.nan
            yc_control_mean = np.nanmean(yc_control) if len(yc_control) > 0 else np.nan

            # Print summary line
            leaf_id = id(leaf_node)
            print(
                f"{leaf_id:<8} {n_samples:<8} {tauF:<10.4f} {tauC:<10.4f} {U_avg:<10.3f} {N_avg:<10.3f} {S_avg:<10.3f}"
            )

            # Store for detailed analysis
            segment_summary.append(
                {
                    "segment_id": leaf_id,
                    "n_samples": n_samples,
                    "tauF": tauF,
                    "tauC": tauC,
                    "U_avg": U_avg,
                    "N_avg": N_avg,
                    "S_avg": S_avg,
                    "conv_treat": conv_treat,
                    "conv_control": conv_control,
                    "yc_treat": yc_treat_mean,
                    "yc_control": yc_control_mean,
                }
            )

    return leaf_nodes, segment_summary


def plot_tree(tree):
    """
    Plot the divergence tree.
    """
    print("\n" + "=" * 50)
    print("PLOTTING DIVERGENCE TREE")
    print("=" * 50)

    fig, ax = plot_divergence_tree(
        tree,
        figsize=(15, 8),
        h_spacing=3.5,
        v_spacing=2.5,
        node_width=3.0,
        node_height=1.2,
    )
    plt.title("Divergence Tree Segmentation Results", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("tests/Trial/results/divergence_tree.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)


def main():
    """
    Main function to run the segmentation analysis.
    """
    # Create results directory if it doesn't exist
    import os

    os.makedirs("tests/Trial/results", exist_ok=True)

    # Load and prepare data
    X, T, YF, YC = load_and_prepare_data()

    # Run segmentation analysis
    tree, best_params, best_score = run_segmentation_analysis(X, T, YF, YC, n_trials=20)

    # Analyze segments
    leaf_nodes, segment_summary = analyze_segments(tree, X, T, YF, YC)

    # Plot the tree
    plot_tree(tree)

    # Save results
    results = {
        "tree": tree,
        "best_params": best_params,
        "best_score": best_score,
        "leaf_nodes": leaf_nodes,
        "segment_summary": segment_summary,
        "X_shape": X.shape,
        "n_segments": len(segment_summary),
    }

    import pickle

    with open("tests/Trial/results/segmentation_results.pickle", "wb") as f:
        pickle.dump(results, f)

    print(f"\nResults saved to tests/Trial/results/")
    print("Files created:")
    print("  - divergence_tree.png")
    print("  - segmentation_results.pickle")

    # Keep plots open
    print("\nTree plot is displayed. Close the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
