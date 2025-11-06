"""
Data Generator for Comparison Simulation

Generates synthetic data with:
- N users with F features (uniformly distributed 0-10)
- Random 50/50 treatment assignment
- Firm outcome (binary) with treatment effects based on X1, X2
- User outcome (continuous) with treatment effects based on X1, X2
- Specific treatment effect structure as specified
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


def generate_comparison_data(
    n_users: int = 10000,
    n_features: int = 10,
    base_subscription_prob: float = 0.5,
    user_outcome_noise_std: float = 1.0,
    random_seed: Optional[int] = None,
    firm_effect_strength: float = 0.2,
    user_effect_strength: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data for comparison simulation.

    Args:
        n_users: Number of users to generate
        n_features: Number of features per user (must be >= 4)
        base_subscription_prob: Base probability of subscription (without treatment)
        user_outcome_noise_std: Standard deviation of user outcome noise
        random_seed: Random seed for reproducibility
        firm_effect_strength: Direct strength of firm treatment effects (default 0.2)
        user_effect_strength: Direct strength of user treatment effects (default 2.0)

    Returns:
        X: Feature matrix (n_users, n_features)
        T: Treatment assignment (n_users,) - binary 0/1
        YF: Firm outcome (n_users,) - binary 0/1 (subscription)
        YC: User outcome (n_users,) - continuous, NaN where YF=0
        tauF: Firm treatment effects (n_users,)
        tauC: User treatment effects (n_users,)
    """

    if n_features < 4:
        raise ValueError("n_features must be >= 4")

    # Set random seed
    rng = np.random.default_rng(random_seed)

    # Generate features: uniform distribution between 0 and 10
    X = rng.uniform(0, 10, size=(n_users, n_features))

    # Generate treatment assignment (random 50/50)
    T = rng.binomial(1, 0.5, size=n_users)

    # Calculate treatment effects based on X1, X2 with direct strength values
    tauF = calculate_firm_treatment_effects(X, firm_effect_strength)
    tauC = calculate_user_treatment_effects(X, user_effect_strength)

    # Generate firm outcomes (subscription)
    # Base probability + treatment effect (only for treated users)
    firm_treatment_effect = tauF * T
    subscription_prob = np.clip(
        base_subscription_prob + firm_treatment_effect, 0.01, 0.99
    )
    YF = rng.binomial(1, subscription_prob, size=n_users)

    # Generate user outcomes (continuous)
    # Mean = 0 + treatment effect (only for treated users) + noise
    user_treatment_effect = tauC * T
    user_outcome_noise = rng.normal(0, user_outcome_noise_std, size=n_users)
    YC_raw = user_treatment_effect + user_outcome_noise

    # Set YC to NaN where YF=0 (non-subscribers)
    YC = np.where(YF == 1, YC_raw, np.nan)

    return X, T, YF, YC, tauF, tauC


def calculate_firm_treatment_effects(
    X: np.ndarray, strength: float = 0.1
) -> np.ndarray:
    """
    Calculate firm outcome treatment effects based on X1, X2.

    Treatment effect structure (using direct strength values):
    0<=X1<5 & 0<=X2<5 : tau_F = -strength
    0<=X1<5 & 5<=X2<=10 : tau_F = +strength
    5<=X1<9 & 0<=X2<=5 : tau_F = -strength
    5<=X1<9 & 5<=X2<=10 : tau_F = +strength
    9<=X1<=10 & 0<=X2<=5 : tau_F = +strength
    9<=X1<=10 & 5<=X2<=10 : tau_F = +strength

    Args:
        X: Feature matrix (n_users, n_features)
        strength: Direct strength of treatment effects

    Returns:
        tauF: Firm treatment effects (n_users,)
    """
    n_users = X.shape[0]
    tauF = np.zeros(n_users)

    X1 = X[:, 0]
    X2 = X[:, 1]

    # Define regions based on X1, X2
    region1 = (X1 >= 0) & (X1 < 5) & (X2 >= 0) & (X2 < 5)
    region2 = (X1 >= 0) & (X1 < 5) & (X2 >= 5) & (X2 <= 10)
    region3 = (X1 >= 5) & (X1 <= 10) & (X2 >= 0) & (X2 < 1)
    region4 = (X1 >= 5) & (X1 <= 10) & (X2 >= 1) & (X2 <= 5)
    region5 = (X1 >= 5) & (X1 <= 10) & (X2 >= 5) & (X2 < 9)
    region6 = (X1 >= 5) & (X1 <= 10) & (X2 >= 9) & (X2 <= 10)

    # Assign treatment effects with direct strength values
    tauF[region1] = -strength
    tauF[region2] = +strength
    tauF[region3] = -strength
    tauF[region4] = -strength
    tauF[region5] = +strength
    tauF[region6] = +strength

    return tauF


def calculate_user_treatment_effects(
    X: np.ndarray, strength: float = 1.0
) -> np.ndarray:
    """
    Calculate user outcome treatment effects based on X1, X2.

    Treatment effect structure (using direct strength values):
    0<=X1<5 & 0<=X2<5 : tau_C = -strength
    0<=X1<5 & 5<=X2<=10 : tau_C = -strength
    5<=X1<9 & 0<=X2<=5 : tau_C = +strength
    5<=X1<9 & 5<=X2<=10 : tau_C = +strength
    9<=X1<=10 & 0<=X2<=5 : tau_C = +strength
    9<=X1<=10 & 5<=X2<=10 : tau_C = -strength

    Args:
        X: Feature matrix (n_users, n_features)
        strength: Direct strength of treatment effects

    Returns:
        tauC: User treatment effects (n_users,)
    """
    n_users = X.shape[0]
    tauC = np.zeros(n_users)

    X1 = X[:, 0]
    X2 = X[:, 1]

    # Define regions based on X1, X2
    region1 = (X1 >= 0) & (X1 < 5) & (X2 >= 0) & (X2 < 5)
    region2 = (X1 >= 0) & (X1 < 5) & (X2 >= 5) & (X2 <= 10)
    region3 = (X1 >= 5) & (X1 <= 10) & (X2 >= 0) & (X2 < 1)
    region4 = (X1 >= 5) & (X1 <= 10) & (X2 >= 1) & (X2 <= 5)
    region5 = (X1 >= 5) & (X1 <= 10) & (X2 >= 5) & (X2 < 9)
    region6 = (X1 >= 5) & (X1 <= 10) & (X2 >= 9) & (X2 <= 10)

    # Assign treatment effects with direct strength values
    tauC[region1] = -strength
    tauC[region2] = -strength
    tauC[region3] = -strength
    tauC[region4] = +strength
    tauC[region5] = +strength
    tauC[region6] = -strength

    return tauC


def get_data_summary(
    X: np.ndarray, T: np.ndarray, YF: np.ndarray, YC: np.ndarray
) -> Dict[str, Any]:
    """
    Get summary statistics of the generated data.

    Args:
        X: Feature matrix
        T: Treatment assignment
        YF: Firm outcome
        YC: User outcome

    Returns:
        Dictionary with summary statistics
    """
    n_users = X.shape[0]
    n_treated = T.sum()
    n_control = n_users - n_treated

    # Firm outcome statistics
    firm_treated_rate = YF[T == 1].mean() if n_treated > 0 else 0
    firm_control_rate = YF[T == 0].mean() if n_control > 0 else 0
    firm_ate = firm_treated_rate - firm_control_rate

    # User outcome statistics (only for subscribers)
    user_treated_mean = np.nanmean(YC[T == 1]) if n_treated > 0 else np.nan
    user_control_mean = np.nanmean(YC[T == 0]) if n_control > 0 else np.nan
    user_ate = (
        user_treated_mean - user_control_mean
        if not (np.isnan(user_treated_mean) or np.isnan(user_control_mean))
        else np.nan
    )

    return {
        "n_users": n_users,
        "n_treated": n_treated,
        "n_control": n_control,
        "treatment_rate": n_treated / n_users,
        "firm_treated_rate": firm_treated_rate,
        "firm_control_rate": firm_control_rate,
        "firm_ate": firm_ate,
        "user_treated_mean": user_treated_mean,
        "user_control_mean": user_control_mean,
        "user_ate": user_ate,
        "subscription_rate": YF.mean(),
        "conversion_rate": np.nanmean(YC[YF == 1]) if YF.sum() > 0 else np.nan,
    }


if __name__ == "__main__":
    # Example usage
    print("Generating comparison data...")

    X, T, YF, YC = generate_comparison_data(
        n_users=10000,
        n_features=10,
        base_subscription_prob=0.5,
        user_outcome_std=1.0,
        random_seed=42,
    )

    print(
        f"Generated data shape: X={X.shape}, T={T.shape}, YF={YF.shape}, YC={YC.shape}"
    )

    # Get summary statistics
    summary = get_data_summary(X, T, YF, YC)
    print("\nData Summary:")
    for key, value in summary.items():
        print(
            f"  {key}: {value:.4f}"
            if isinstance(value, (int, float))
            else f"  {key}: {value}"
        )
