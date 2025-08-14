"""
Tuning utilities for DivergenceTree.

Assumptions:
- Randomized treatment assignment (or unconfoundedness within node).
- Firm outcome YF is Bernoulli; consumer outcome YC is numeric and only defined when YF==1.
- Validation loss is standardized by validation SEs so firm/consumer contribute fairly.
- Adds a stump penalty (discourages trivial trees).
- Adds a lexicographic tie-break toward stronger convergence/divergence on validation.

Public API:
- tune_with_optuna_partial(...)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from .tree import DivergenceTree


def _estimate_effects_simple(T, YF, YC, idx):
    """Diff-in-means estimates for firm (YF) and consumer (YC among converters)."""
    sub_T, sub_YF, sub_YC = T[idx], YF[idx], YC[idx]
    treated = sub_T == 1
    control = ~treated
    if treated.sum() == 0 or control.sum() == 0:
        return np.nan, np.nan, idx.size

    # Firm-side TE (Bernoulli conversion)
    tauF = sub_YF[treated].mean() - sub_YF[control].mean()

    # Consumer-side TE among converters in each arm
    conv_treat = treated & (sub_YF == 1)
    conv_ctrl = control & (sub_YF == 1)
    if conv_treat.sum() == 0 or conv_ctrl.sum() == 0:
        tauC = np.nan
    else:
        yC1 = sub_YC[conv_treat]
        yC1 = yC1[np.isfinite(yC1)]
        yC0 = sub_YC[conv_ctrl]
        yC0 = yC0[np.isfinite(yC0)]
        tauC = (yC1.mean() - yC0.mean()) if (yC1.size > 0 and yC0.size > 0) else np.nan
    return tauF, tauC, idx.size


# --------------------------- Validation SEs (standardization) ---------------------------


def _leaf_se_F_bernoulli(T, YF, idx) -> float:
    """
    SE for diff-in-means of YF (Bernoulli):
      SE^2 = p1(1-p1)/n1 + p0(1-p0)/n0
    """
    sub_T, sub_YF = T[idx], YF[idx]
    treated = sub_T == 1
    control = ~treated
    n1, n0 = int(treated.sum()), int(control.sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    p1 = float(sub_YF[treated].mean())
    p0 = float(sub_YF[control].mean())
    var1 = p1 * (1 - p1)
    var0 = p0 * (1 - p0)
    se2 = var1 / max(n1, 1) + var0 / max(n0, 1)
    return float(np.sqrt(max(se2, 1e-8)))


def _leaf_se_C_generic(T, YF, YC, idx) -> float:
    """
    SE for diff-in-means of YC among converters:
      SE^2 = Var(YC|T=1, YF=1)/c1 + Var(YC|T=0, YF=1)/c0
    """
    sub_T, sub_YF, sub_YC = T[idx], YF[idx], YC[idx]
    treated = sub_T == 1
    control = ~treated
    conv_treat = treated & (sub_YF == 1)
    conv_ctrl = control & (sub_YF == 1)
    c1, c0 = int(conv_treat.sum()), int(conv_ctrl.sum())
    if c1 == 0 or c0 == 0:
        return np.nan
    v1 = float(np.var(sub_YC[conv_treat], ddof=1)) if c1 > 1 else 0.0
    v0 = float(np.var(sub_YC[conv_ctrl], ddof=1)) if c0 > 1 else 0.0
    se2 = (v1 / max(c1, 1)) + (v0 / max(c0, 1))
    return float(np.sqrt(max(se2, 1e-8)))


# --------------------------- Standardized validation loss ---------------------------


def evaluate_tree_consistency(
    tree: DivergenceTree,
    X_val: np.ndarray,
    T_val: np.ndarray,
    YF_val: np.ndarray,
    YC_val: np.ndarray,
    nan_penalty: float = 1.0,
    weight_by: str = "val",  # "val" or "both"
) -> float:
    """
    Standardized stability loss across leaves (lower is better):

      sum_leaves w * [ ((tauF_val - tauF_train)/SE_F_val)^2
                     + ((tauC_val - tauC_train)/SE_C_val)^2 ]

    If a leaf's validation TE or SE is undefined, we add w * nan_penalty.
    """
    leaves_val = tree.predict_leaf(X_val)
    uniq_leaves = list({id(l): l for l in leaves_val}.values())

    n_val_total = X_val.shape[0]
    n_tr_total = sum(int(l.n or 0) for l in uniq_leaves)

    loss = 0.0
    for leaf in uniq_leaves:
        m_val = np.array([l is leaf for l in leaves_val])
        n_val_leaf = int(m_val.sum())
        if n_val_leaf == 0:
            continue

        idx_val = np.where(m_val)[0]
        tF_val, tC_val, _ = _estimate_effects_simple(T_val, YF_val, YC_val, idx_val)
        tF_tr = leaf.tauF
        tC_tr = leaf.tauC

        # leaf weight
        if weight_by == "val":
            w = n_val_leaf / max(1, n_val_total)
        else:  # "both"
            w = (int(leaf.n or 0) + n_val_leaf) / max(1, n_tr_total + n_val_total)

        # validation SEs for standardization
        seF = _leaf_se_F_bernoulli(T_val, YF_val, idx_val)
        seC = _leaf_se_C_generic(T_val, YF_val, YC_val, idx_val)

        ok = (
            np.isfinite(tF_val)
            and np.isfinite(tC_val)
            and np.isfinite(tF_tr)
            and np.isfinite(tC_tr)
            and np.isfinite(seF)
            and np.isfinite(seC)
            and seF > 0
            and seC > 0
        )
        if not ok:
            loss += w * float(nan_penalty)
            continue

        zF = (float(tF_val) - float(tF_tr)) / max(seF, 1e-8)
        zC = (float(tC_val) - float(tC_tr)) / max(seC, 1e-8)
        loss += w * (zF**2 + zC**2)

    return float(loss)


# --------------------------- Validation co-movement bonus ---------------------------


def _phi(d: float, mode: str) -> float:
    """Same φ as training objective (mode: 'both' | 'converge' | 'diverge')."""
    mode = (mode or "both").lower()
    if mode == "converge":
        return max(0.0, d)
    if mode == "diverge":
        return max(0.0, -d)
    return abs(d)  # "both"


def validation_comovement_bonus(
    tree: DivergenceTree,
    X_val: np.ndarray,
    T_val: np.ndarray,
    YF_val: np.ndarray,
    YC_val: np.ndarray,
) -> float:
    """
    Measures targeted co-movement on validation (higher is better), standardized by
    validation SEs and weighted by validation leaf sizes.
    """
    leaves_val = tree.predict_leaf(X_val)
    uniq_leaves = list({id(l): l for l in leaves_val}.values())
    n_val_total = X_val.shape[0]
    mode = getattr(tree, "co_movement", "both")

    bonus = 0.0
    for leaf in uniq_leaves:
        m_val = np.array([l is leaf for l in leaves_val])
        idx_val = np.where(m_val)[0]
        if idx_val.size == 0:
            continue

        # validation effects & SEs
        tF_val, tC_val, _ = _estimate_effects_simple(T_val, YF_val, YC_val, idx_val)
        seF = _leaf_se_F_bernoulli(T_val, YF_val, idx_val)
        seC = _leaf_se_C_generic(T_val, YF_val, YC_val, idx_val)

        ok = (
            np.isfinite(tF_val)
            and np.isfinite(tC_val)
            and np.isfinite(seF)
            and np.isfinite(seC)
            and seF > 0
            and seC > 0
        )
        if not ok:
            continue

        # center on the leaf's learned effects (training promise)
        zF = (float(tF_val) - float(leaf.tauF)) / max(seF, 1e-8)
        zC = (float(tC_val) - float(leaf.tauC)) / max(seC, 1e-8)
        d = zF * zC

        w = idx_val.size / max(1, n_val_total)
        bonus += w * _phi(d, mode)

    return float(bonus)


# --------------------------- Optuna tuning (with penalties & tie-break) ---------------------------


def tune_with_optuna_partial(
    X: np.ndarray,
    T: np.ndarray,
    YF: np.ndarray,
    YC: np.ndarray,
    fixed: Optional[Dict[str, Any]] = None,
    search_space: Optional[Dict[str, Dict[str, Any]]] = None,
    valid_fraction: float = 0.25,
    n_trials: int = 50,
    random_state: Optional[int] = 123,  # reproducibility only
    stump_penalty: float = 1.0,  # added to loss if the tree makes no split
    tie_break_eps: float = 1e-6,  # tiny factor for lexicographic tie-break
):
    import optuna

    fixed = dict(fixed or {})
    search_space = dict(search_space or {})

    # --- guard: never allow random_state in search space ---
    if "random_state" in search_space:
        raise ValueError("Do not include 'random_state' in search_space.")

    # --- split (reproducible, not tunable) ---
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1 - valid_fraction))
    train_idx, val_idx = idx[:cut], idx[cut:]

    X_tr, X_val = X[train_idx], X[val_idx]
    T_tr, T_val = T[train_idx], T[val_idx]
    YF_tr, YF_val = YF[train_idx], YF[val_idx]
    YC_tr, YC_val = YC[train_idx], YC[val_idx]

    def suggest_params(trial):
        """
        Build params from search_space. Supports:
          - {"type":"float","low":a,"high":b,"log":True/False}
          - {"type":"int","low":a,"high":b,"step":k}
          - {"type":"cat","choices":[...]}
        """
        params = dict(fixed)
        for name, spec in search_space.items():
            t = spec["type"]
            if t == "float":
                if spec.get("log", False):
                    params[name] = trial.suggest_float(
                        name, spec["low"], spec["high"], log=True
                    )
                else:
                    params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif t == "int":
                step = spec.get("step", 1)
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"], step=step
                )
            elif t == "cat":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown type in search_space for {name}: {t}")
        return params

    def objective(trial):
        params = suggest_params(trial)
        tree = DivergenceTree(
            random_state=fixed.get("random_state", random_state),
            **{k: v for k, v in params.items() if k != "random_state"},
        )
        try:
            tree.fit(X_tr, T_tr, YF_tr, YC_tr)

            # 1) Base: standardized consistency loss (lower is better)
            base_loss = evaluate_tree_consistency(
                tree,
                X_val,
                T_val,
                YF_val,
                YC_val,
                nan_penalty=1.0,
                weight_by="val",  # or "both"
            )

            # 2) Stump penalty (discourage trivial trees)
            is_stump = tree.root_.feature is None
            base_loss += stump_penalty if is_stump else 0.0

            # 3) Lexicographic tie-break toward stronger (di)vergence on validation
            #    We subtract a tiny epsilon * bonus so primary optimization remains the loss.
            B = validation_comovement_bonus(tree, X_val, T_val, YF_val, YC_val)
            loss = base_loss - (tie_break_eps * B)

            return loss if np.isfinite(loss) else 1e6

        except Exception:
            # Any failure → large loss so Optuna moves on
            return 1e6

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    tuned = {k: v for k, v in study.best_trial.params.items() if k != "random_state"}
    best_params = dict(fixed)
    best_params.update(tuned)
    best_loss = study.best_value  # lower is better
    return best_params, best_loss
