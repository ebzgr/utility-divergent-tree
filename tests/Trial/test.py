import numpy as np
import matplotlib.pyplot as plt
from datagenerating import generate_data
from helper import (
    plot_latent_overview,
    plot_P_and_pi_overview,
    print_summary,
    make_group_table_from_arrays,
    plot_six_panels_outcomes_and_pi,
)
from typing import Dict, List, Union, Optional, Tuple
import pandas as pd
import pdb

# ---------------------- Example configuration ----------------------
n = 100000
p = 20
seed = 123

# Fix gate indices for interpretability/visualization
gate_indices = {"U": 0, "N": 1, "S": 2}  # visualize along X[:,0], X[:,1], X[:,2]

# Try hard gates first (switch to "soft" to test logistic gates)
gate_type = "hard"
gate_thresholds = {"U": 0.0, "N": 0.0, "S": 0.0}
gate_lambdas = {"U": 5.0, "N": 5.0, "S": 5.0}  # used only for soft gates

# Model knobs
v = 1.0  # noise scale for latent scores
kappa = 1.0  # post-score sharpness (polarization)
m_U = 1.0  # GLOBAL bipolarity strength (offset size)
m_N = 1.0  # GLOBAL bipolarity strength (offset size)
m_S = 1.0  # GLOBAL bipolarity strength (offset size)
a_u = 0.4  # P_short = N + a_u * U
a_s = 0.8  # P_long  = U + a_s * S
a0 = -0.7  # subscription: pi = sigmoid(a0 + a_p * P)
a_p = 1

# ---------------------- Generate synthetic data ----------------------
X, L, T, YF, YC, P_short, P_long, pi = generate_data(
    n=n,
    p=p,
    v=v,
    kappa=kappa,
    m_U=m_U,
    m_N=m_N,
    m_S=m_S,
    gate_type=gate_type,
    gate_indices=gate_indices,
    gate_thresholds=gate_thresholds,
    gate_lambdas=gate_lambdas,
    a_u=a_u,
    a_s=a_s,
    a0=a0,
    a_p=a_p,
    seed=seed,
)

# Create comprehensive DataFrame with all data
df = pd.DataFrame(
    {
        # Features (first 10 columns for brevity, you can adjust)
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

df.to_pickle("tests/Trial/data/sim3.pickle")

# threshold = 0.5
# df[(df.S > threshold) & (df.N < 0.1) & (df.YF == 1) & (df.Treat == 0)].U.hist()
# df[(df.S > threshold) & (df.N < 0.1) & (df.YF == 1) & (df.Treat == 1)].U.hist()
# df[(df.S > threshold) & (df.N < 0.1) & (df.YF == 1)].groupby("Treat").U.mean()
# Correlation matrix for latent variables
latent_df = pd.DataFrame({"U": L["U"], "N": L["N"], "S": L["S"]})
print("\nLatent variable correlations:")
print(latent_df.corr())

# plot_latent_overview(X, L, gate_indices, nbins=20, figsize=(16, 8), show=True)

# New overview (P and pi relationships):
# plot_P_and_pi_overview(L, T, P_short, P_long, pi, nbins=20, figsize=(18, 9), show=True)

# Quick summary:
print_summary(YF, T, L)

table = make_group_table_from_arrays(T, YF, YC, L["U"], L["N"], L["S"])
print(table)
fig, axes = plot_six_panels_outcomes_and_pi(T, YF, YC, pi, L, nbins=20)
# pdb.set_trace()
