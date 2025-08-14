import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import pdb


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out


def _draw_weights(p: int, rng: np.random.Generator) -> np.ndarray:
    # Draw and L2-normalize
    w = rng.normal(0.0, 1.0, size=p)
    norm = np.linalg.norm(w)
    if norm == 0:
        return w
    return w / norm


def _gate_value(
    X: np.ndarray,
    indices: Union[int, List[int]],
    threshold: float,
    gate_type: str,
    lam: float,
) -> np.ndarray:
    """
    Returns gate value g(x):
      - hard: in {0,1}
      - soft: in (0,1), logistic around threshold with sharpness lam
    Gate operates on the mean of selected features.
    """
    if isinstance(indices, int):
        z = X[:, indices]
    else:
        z = X[:, indices].mean(axis=1)
    if gate_type == "hard":
        g = (z > threshold).astype(float)
    elif gate_type == "soft":
        g = _sigmoid(lam * (z - threshold))
    else:
        raise ValueError("gate_type must be 'hard' or 'soft'")
    return g


def _latent_construct(
    X: np.ndarray,
    w: np.ndarray,
    g: np.ndarray,
    m: float,
    v: float,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    k* = (2g-1)*m + Xw + v*eps, then squash: k = sigmoid(kappa * k*)
    """
    eps = rng.normal(0.0, 1.0, size=X.shape[0])
    k_star = (2.0 * g - 1.0) * m + X @ w + v * eps
    return _sigmoid(kappa * k_star)


def generate_data(
    n: int = 50_000,
    p: int = 100,
    v: float = 1.0,  # noise knob (higher => noisier latent constructs)
    kappa: float = 1.0,  # optional sharpness after latent score (polarization)
    m_U: float = 1.0,  # X-anchored bipolarity strength for U
    m_N: float = 1.0,  # ... for N
    m_S: float = 1.0,  # ... for S
    gate_type: str = "hard",  # 'hard' or 'soft' (logit gate)
    gate_indices: Optional[
        Dict[str, Union[int, List[int]]]
    ] = None,  # feature index/indices per construct
    gate_thresholds: Optional[
        Dict[str, float]
    ] = None,  # threshold per construct (default 0.0)
    gate_lambdas: Optional[
        Dict[str, float]
    ] = None,  # soft gate sharpness per construct (default 5.0)
    a_u: float = 0.3,  # P_short = logaddexp(N, a_u * U)
    a_s: float = 0.9,  # P_long  = logaddexp(S, a_s * U)
    a0: float = 0.0,  # subscription: pi = a0 + a_p * P
    a_p: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate (X, L, T, YF, YC) with:
      - X: (n,p) i.i.d. N(0,1)
      - Latents L={'U','N','S'} in [0,1], each constructed via X-anchored gate (hard/soft)
      - T: treatment (0=Short, 1=Long), Bernoulli(0.5)
      - YF: subscription, Bernoulli(pi) with pi = a0 + a_p * P
            where P_short = logaddexp(N, a_u*U), P_long = logaddexp(S, a_s*U)
      - YC: consumer outcome = U observed only if YF==1, else NaN
    """
    rng = np.random.default_rng(seed)

    # 1) Features
    X = rng.normal(0.0, 1.0, size=(n, p))

    # 2) Draw weights for latent linear components
    w_U = _draw_weights(p, rng)
    w_N = _draw_weights(p, rng)
    w_S = _draw_weights(p, rng)

    # 3) Gate config (indices, thresholds, lambdas)
    if gate_indices is None:
        # pick one random feature per construct (distinct if possible)
        idxs = (
            rng.choice(p, size=3, replace=False)
            if p >= 3
            else rng.integers(0, p, size=3)
        )
        gate_indices = {"U": int(idxs[0]), "N": int(idxs[1]), "S": int(idxs[2])}
    else:
        gate_indices = dict(gate_indices)  # shallow copy

    if gate_thresholds is None:
        gate_thresholds = {"U": 0.0, "N": 0.0, "S": 0.0}
    else:
        gate_thresholds = dict(gate_thresholds)

    if gate_lambdas is None:
        gate_lambdas = {"U": 5.0, "N": 5.0, "S": 5.0}
    else:
        gate_lambdas = dict(gate_lambdas)

    # 4) Compute gate values per construct
    gU = _gate_value(
        X,
        gate_indices["U"],
        gate_thresholds.get("U", 0.0),
        gate_type,
        gate_lambdas.get("U", 5.0),
    )
    gN = _gate_value(
        X,
        gate_indices["N"],
        gate_thresholds.get("N", 0.0),
        gate_type,
        gate_lambdas.get("N", 5.0),
    )
    gS = _gate_value(
        X,
        gate_indices["S"],
        gate_thresholds.get("S", 0.0),
        gate_type,
        gate_lambdas.get("S", 5.0),
    )

    # 5) Latent continuous constructs in [0,1]
    U = _latent_construct(X, w_U, gU, m_U, v, kappa, rng)
    N = _latent_construct(X, w_N, gN, m_N, v, kappa, rng)
    S = _latent_construct(X, w_S, gS, m_S, v, kappa, rng)
    L = {"U": U, "N": N, "S": S}

    # 6) Treatment (Short/Long)
    T = rng.binomial(1, 0.5, size=n).astype(int)  # 0=Short, 1=Long

    # 7) Perceived usefulness score (log-sum-exp softmax-like decision)
    # For short trials: subscribe if Novelty is high OR if weighted Usefulness is high
    P_short = np.logaddexp(N, a_u * U)

    # For long trials: subscribe if Sunk Cost is high OR if weighted Usefulness is high
    P_long = np.logaddexp(U, a_s * S)

    P = np.where(T == 0, P_short, P_long)

    # 8) Subscription probability and outcome
    # Direct probability calculation since P values are already in [0,1]
    pi = a0 + a_p * P
    YF = (rng.random(size=n) < pi).astype(int)

    # 9) Consumer outcome (observed only if subscribed)
    YC = np.full(n, np.nan, dtype=float)
    YC[YF == 1] = U[YF == 1]

    return X, L, T, YF, YC, P_short, P_long, pi
