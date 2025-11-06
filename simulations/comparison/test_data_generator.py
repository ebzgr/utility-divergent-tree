"""
Simple test script for the data generator with visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_comparison_data, get_data_summary


def test_data_generator():
    """Test the data generator with a small sample and create visualizations."""

    print("Testing data generator...")

    # Generate small sample with stronger treatment effects
    X, T, YF, YC, tauF, tauC = generate_comparison_data(
        n_users=1000,
        n_features=10,
        base_subscription_prob=0.5,
        user_outcome_noise_std=1.0,
        random_seed=42,
        firm_effect_strength=0.3,  # Direct firm effect strength
        user_effect_strength=3.0,  # Direct user effect strength
    )

    print(f"Data shapes:")
    print(f"  X: {X.shape}")
    print(f"  T: {T.shape}")
    print(f"  YF: {YF.shape}")
    print(f"  YC: {YC.shape}")
    print(f"  tauF: {tauF.shape}")
    print(f"  tauC: {tauC.shape}")

    print(f"\nBasic statistics:")
    print(f"  Treatment rate: {T.mean():.3f}")
    print(f"  Subscription rate: {YF.mean():.3f}")
    print(f"  Firm effects range: [{tauF.min():.3f}, {tauF.max():.3f}]")
    print(f"  User effects range: [{tauC.min():.3f}, {tauC.max():.3f}]")

    # Check treatment effect structure
    print(f"\nTreatment effect structure verification:")
    print(f"  Unique firm effects: {np.unique(tauF)}")
    print(f"  Unique user effects: {np.unique(tauC)}")

    # Get summary
    summary = get_data_summary(X, T, YF, YC)
    print(f"\nData summary:")
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Create visualizations
    create_visualizations(X, T, YF, YC, tauF, tauC)

    # Also print analysis
    analyze_treatment_effects(X, YF, YC, tauF, tauC)

    print("\nData generator test completed successfully!")


def create_visualizations(X, T, YF, YC, tauF, tauC):
    """Create 2D scatter plots showing firm and user outcomes over X1, X2."""

    # Extract X1 and X2
    X1 = X[:, 0]
    X2 = X[:, 1]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Firm Outcome (Subscription)
    # Orange for subscription, Blue for no subscription
    subscription_mask = YF == 1
    no_subscription_mask = YF == 0

    ax1.scatter(
        X1[no_subscription_mask],
        X2[no_subscription_mask],
        c="blue",
        alpha=0.6,
        s=20,
        label="No Subscription",
    )
    ax1.scatter(
        X1[subscription_mask],
        X2[subscription_mask],
        c="orange",
        alpha=0.6,
        s=20,
        label="Subscription",
    )

    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")
    ax1.set_title("Firm Outcome (Subscription)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: User Outcome (only for subscribers)
    # Green shades based on user outcome value
    subscriber_mask = YF == 1
    if subscriber_mask.sum() > 0:
        # Get user outcomes for subscribers only
        user_outcomes = YC[subscriber_mask]
        X1_sub = X1[subscriber_mask]
        X2_sub = X2[subscriber_mask]

        # Create color map based on user outcome values
        scatter = ax2.scatter(
            X1_sub, X2_sub, c=user_outcomes, cmap="Greens", alpha=0.7, s=20
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("User Outcome")

        ax2.set_xlabel("X1")
        ax2.set_ylabel("X2")
        ax2.set_title("User Outcome (Subscribers Only)")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No Subscribers",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("User Outcome (No Subscribers)")

    plt.tight_layout()
    plt.savefig("data_generator_test_plots.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Visualizations saved as 'data_generator_test_plots.png'")


def analyze_treatment_effects(X, YF, YC, tauF, tauC):
    """Analyze treatment effects by region when matplotlib is not available."""

    X1 = X[:, 0]
    X2 = X[:, 1]

    print("\nTreatment Effect Analysis by Region:")
    print("=" * 50)

    # Define regions
    regions = [
        ("0<=X1<5 & 0<=X2<5", (X1 >= 0) & (X1 < 5) & (X2 >= 0) & (X2 < 5)),
        ("0<=X1<5 & 5<=X2<=10", (X1 >= 0) & (X1 < 5) & (X2 >= 5) & (X2 <= 10)),
        ("5<=X1<9 & 0<=X2<5", (X1 >= 5) & (X1 < 9) & (X2 >= 0) & (X2 < 5)),
        ("5<=X1<9 & 5<=X2<=10", (X1 >= 5) & (X1 < 9) & (X2 >= 5) & (X2 <= 10)),
        ("9<=X1<=10 & 0<=X2<5", (X1 >= 9) & (X1 <= 10) & (X2 >= 0) & (X2 < 5)),
        ("9<=X1<=10 & 5<=X2<=10", (X1 >= 9) & (X1 <= 10) & (X2 >= 5) & (X2 <= 10)),
    ]

    for region_name, region_mask in regions:
        n_region = region_mask.sum()
        if n_region > 0:
            subscription_rate = YF[region_mask].mean()
            user_outcome_mean = np.nanmean(YC[region_mask])
            firm_effect = tauF[region_mask][0]  # Should be same for all in region
            user_effect = tauC[region_mask][0]  # Should be same for all in region

            print(f"\n{region_name}:")
            print(f"  Count: {n_region}")
            print(f"  Subscription rate: {subscription_rate:.3f}")
            print(f"  User outcome mean: {user_outcome_mean:.3f}")
            print(f"  Firm effect: {firm_effect:.3f}")
            print(f"  User effect: {user_effect:.3f}")


if __name__ == "__main__":
    test_data_generator()
