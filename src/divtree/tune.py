import numpy as np
from typing import Dict, Any, Optional, Tuple
from .tree import DivergenceTree


def _estimate_effects_simple(T, YF, YC, idx):
    sub_T, sub_YF, sub_YC = T[idx], YF[idx], YC[idx]
    treated = sub_T == 1
    control = ~treated
    if treated.sum() == 0 or control.sum() == 0:
        return np.nan, np.nan, idx.size
    tauF = sub_YF[treated].mean() - sub_YF[control].mean()
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


def _joint_score(lambda_, tauF_root, tauC_root, leaf_taus):
    score = 0.0
    for w, tF, tC in leaf_taus:
        if not (
            np.isfinite(tF)
            and np.isfinite(tC)
            and np.isfinite(tauF_root)
            and np.isfinite(tauC_root)
        ):
            continue
        score += w * (
            (tF - tauF_root) ** 2 + (tC - tauC_root) ** 2 + lambda_ * abs(tF * tC)
        )
    return float(score)


def evaluate_tree_on_validation(
    tree: DivergenceTree,
    X_val: np.ndarray,
    T: np.ndarray,
    YF: np.ndarray,
    YC: np.ndarray,
) -> float:
    idx_val = np.arange(X_val.shape[0])
    tauF_root, tauC_root, n_root = _estimate_effects_simple(T, YF, YC, idx_val)
    leaves = tree.predict_leaf(X_val)
    leaf_scores = []
    for leaf in set(leaves):
        mask = np.array([l is leaf for l in leaves])
        idx = idx_val[mask]
        if idx.size == 0:
            continue
        tF, tC, n = _estimate_effects_simple(T, YF, YC, idx)
        leaf_scores.append((idx.size / n_root, tF, tC))
    return _joint_score(tree.lambda_, tauF_root, tauC_root, leaf_scores)


def tune_with_optuna_partial(
    X: np.ndarray,
    T: np.ndarray,
    YF: np.ndarray,
    YC: np.ndarray,
    fixed: Optional[Dict[str, Any]] = None,
    search_space: Optional[Dict[str, Dict[str, Any]]] = None,
    valid_fraction: float = 0.25,
    n_trials: int = 50,
    random_state: Optional[int] = 123,  # <- used only for reproducibility
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
        params = dict(fixed)
        for name, spec in search_space.items():
            t = spec["type"]
            if t == "float":
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
        # ensure random_state is fixed (not from params)
        tree = DivergenceTree(
            random_state=fixed.get("random_state", random_state),
            **{k: v for k, v in params.items() if k != "random_state"},
        )
        try:
            tree.fit(X_tr, T_tr, YF_tr, YC_tr)
            score = evaluate_tree_on_validation(tree, X_val, T_val, YF_val, YC_val)
            return -score if np.isfinite(score) else 1e6  # Optuna minimizes
        except Exception:
            return 1e6

    # Seed Optuna's sampler for reproducibility (not tuned)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Build best_params: fixed + tuned (explicitly drop any random_state just in case)
    tuned = {k: v for k, v in study.best_trial.params.items() if k != "random_state"}
    best_params = dict(fixed)
    best_params.update(tuned)
    best_score = -study.best_value
    return best_params, best_score
