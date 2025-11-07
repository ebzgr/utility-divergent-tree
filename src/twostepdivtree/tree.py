"""
Two-Step Divergence Tree using Causal Forest and Classification Tree.

This implementation uses a two-step approach:
1. Step 1: Use causal forest to estimate treatment effects for YF and YC separately
2. Step 2: Categorize observations into 4 region types, then train a classification tree

The algorithm:
1. Fit separate causal forests for firm (YF) and consumer (YC) outcomes
2. Estimate tauF and tauC for each observation
3. Categorize observations into 4 region types based on effect signs:
   - Region 1: tauF > 0 and tauC > 0 (both positive)
   - Region 2: tauF > 0 and tauC <= 0 (firm+, customer-)
   - Region 3: tauF <= 0 and tauC > 0 (firm-, customer+)
   - Region 4: tauF <= 0 and tauC <= 0 (both negative)
4. Train a classification tree to predict region types from features
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import optuna

try:
    from econml.dml import CausalForestDML
except ImportError:
    raise ImportError(
        "econml is required for TwoStepDivergenceTree. "
        "Install it with: pip install econml"
    )


class TwoStepDivergenceTree:
    """
    Two-step divergence tree using causal forest and classification tree.

    This class implements an alternative approach to divergence tree estimation:
    1. Uses causal forest to estimate treatment effects separately for each outcome
    2. Categorizes observations into 4 region types based on effect signs
    3. Trains a classification tree to predict region types

    Parameters
    ----------
    causal_forest_params : dict, optional
        Parameters for causal forest models. Common parameters:
        - n_estimators: int, default=100
        - max_depth: int, default=None
        - min_samples_split: int, default=10
        - min_samples_leaf: int, default=5
        - n_jobs: int, default=None
            Number of parallel jobs to run. None means 1, -1 means use all processors.
            Causal forests support parallelization for faster training.
        - random_state: int, optional
    classification_tree_params : dict, optional
        Parameters for classification tree. Common parameters:
        - max_depth: int, default=None
        - min_samples_split: int, default=2
        - min_samples_leaf: int, default=1
        - random_state: int, optional
        Note: DecisionTreeClassifier (single tree) does not support parallelization.
    causal_forest_tune_params : dict, optional
        Parameters for causal forest tuning. The causal forest will be tuned
        during fit() using econml's built-in tune() method.
        - params: str or dict, default="auto"
            If "auto", uses econml's default grid. Otherwise, provide a dict
            with grid search space: {"param_name": [value1, value2, ...], ...}

    Attributes
    ----------
    causal_forest_F_ : CausalForestDML
        Fitted causal forest for firm outcome (YF).
    causal_forest_C_ : CausalForestDML
        Fitted causal forest for consumer outcome (YC).
    classification_tree_ : DecisionTreeClassifier
        Fitted classification tree for region type prediction.
    tauF_ : np.ndarray
        Estimated treatment effects for firm outcome.
    tauC_ : np.ndarray
        Estimated treatment effects for consumer outcome.
    region_types_ : np.ndarray
        Region type labels (1-4) for training data.
    """

    def __init__(
        self,
        causal_forest_params: Optional[Dict[str, Any]] = None,
        classification_tree_params: Optional[Dict[str, Any]] = None,
        causal_forest_tune_params: Optional[Dict[str, Any]] = None,
    ):
        self.causal_forest_params = dict(causal_forest_params or {})
        self.classification_tree_params = dict(classification_tree_params or {})
        self.causal_forest_tune_params = dict(causal_forest_tune_params or {})

        # Set defaults for causal forest
        if "n_estimators" not in self.causal_forest_params:
            self.causal_forest_params["n_estimators"] = 100
        if "max_depth" not in self.causal_forest_params:
            self.causal_forest_params["max_depth"] = None
        if "min_samples_split" not in self.causal_forest_params:
            self.causal_forest_params["min_samples_split"] = 10
        if "min_samples_leaf" not in self.causal_forest_params:
            self.causal_forest_params["min_samples_leaf"] = 5
        # n_jobs defaults to None (1 job) if not specified - users can set it to -1 for all CPUs

        # Set defaults for classification tree
        if "max_depth" not in self.classification_tree_params:
            self.classification_tree_params["max_depth"] = None
        if "min_samples_split" not in self.classification_tree_params:
            self.classification_tree_params["min_samples_split"] = 2
        if "min_samples_leaf" not in self.classification_tree_params:
            self.classification_tree_params["min_samples_leaf"] = 1

        # Will be set during fit
        self.causal_forest_F_: Optional[CausalForestDML] = None
        self.causal_forest_C_: Optional[CausalForestDML] = None
        self.classification_tree_: Optional[DecisionTreeClassifier] = None
        self.tauF_: Optional[np.ndarray] = None
        self.tauC_: Optional[np.ndarray] = None
        self.region_types_: Optional[np.ndarray] = None
        self._fit_data: Dict[str, Any] = {}

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        YF: np.ndarray,
        YC: np.ndarray,
        auto_tune_classification_tree: bool = False,
        classification_tree_search_space: Optional[Dict[str, Dict[str, Any]]] = None,
        classification_tree_tune_n_trials: int = 30,
        classification_tree_tune_n_splits: int = 5,
    ) -> "TwoStepDivergenceTree":
        """
        Fit the two-step divergence tree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        T : np.ndarray of shape (n_samples,)
            Treatment indicator (0 or 1).
        YF : np.ndarray of shape (n_samples,)
            Firm outcome (binary or continuous, may contain NaN).
        YC : np.ndarray of shape (n_samples,)
            Consumer outcome (binary or continuous, may contain NaN).
        auto_tune_classification_tree : bool, default=False
            If True, automatically tunes the classification tree hyperparameters
            using Optuna. If False, uses the provided `classification_tree_params`.
        classification_tree_search_space : dict, optional
            Search space for classification tree tuning (Optuna format).
            Only used if `auto_tune_classification_tree=True`.
            If None, uses default search space.
            Format: {"param_name": {"low": value, "high": value, "log": bool, "step": int}}
        classification_tree_tune_n_trials : int, default=30
            Number of Optuna trials for classification tree tuning.
            Only used if `auto_tune_classification_tree=True`.
        classification_tree_tune_n_splits : int, default=5
            Number of CV folds for classification tree tuning.
            Only used if `auto_tune_classification_tree=True`.

        Returns
        -------
        self : TwoStepDivergenceTree
            Returns self for method chaining.
        """
        X = np.asarray(X)
        T = np.asarray(T)
        YF = np.asarray(YF)
        YC = np.asarray(YC)

        # Input validation
        n = X.shape[0]
        if len(T) != n or len(YF) != n or len(YC) != n:
            raise ValueError(
                f"Input arrays must have matching lengths: "
                f"X={n}, T={len(T)}, YF={len(YF)}, YC={len(YC)}"
            )

        if not np.all(np.isin(T, [0, 1])):
            raise ValueError("T must be in {0,1}.")

        # Store fit data
        self._fit_data = dict(X=X, T=T, YF=YF, YC=YC)

        # Step 1: Fit causal forests for each outcome with tuning
        print("Fitting and tuning causal forest for firm outcome (YF)...")
        self.causal_forest_F_ = CausalForestDML(**self.causal_forest_params)

        # Handle NaN values for YF
        valid_F = ~np.isnan(YF)
        if valid_F.sum() < 10:
            raise ValueError("Too few valid observations for firm outcome.")

        # Tune causal forest using built-in tune() method
        tune_params = self.causal_forest_tune_params.get("params", "auto")
        self.causal_forest_F_.tune(
            Y=YF[valid_F],
            T=T[valid_F],
            X=X[valid_F],
            params=tune_params,
        )

        # Fit with tuned parameters
        self.causal_forest_F_.fit(
            Y=YF[valid_F],
            T=T[valid_F],
            X=X[valid_F],
        )

        print("Fitting and tuning causal forest for consumer outcome (YC)...")
        self.causal_forest_C_ = CausalForestDML(**self.causal_forest_params)

        # Handle NaN values for YC
        valid_C = ~np.isnan(YC)
        if valid_C.sum() < 10:
            raise ValueError("Too few valid observations for consumer outcome.")

        # Tune causal forest using built-in tune() method
        self.causal_forest_C_.tune(
            Y=YC[valid_C],
            T=T[valid_C],
            X=X[valid_C],
            params=tune_params,
        )

        # Fit with tuned parameters
        self.causal_forest_C_.fit(
            Y=YC[valid_C],
            T=T[valid_C],
            X=X[valid_C],
        )

        # Predict treatment effects for all observations
        print("Predicting treatment effects...")
        self.tauF_ = self.causal_forest_F_.effect(X)
        self.tauC_ = self.causal_forest_C_.effect(X)

        # Step 2: Categorize observations into 4 region types
        print("Categorizing observations into region types...")
        self.region_types_ = self._categorize_region_types(self.tauF_, self.tauC_)

        # Step 3: Train classification tree (with optional auto-tuning)
        if auto_tune_classification_tree:
            print("Auto-tuning classification tree hyperparameters...")
            # Only pass truly fixed parameters (like random_state) as fixed
            # Parameters with None or default values should be tuned
            fixed_params = {}
            if "random_state" in self.classification_tree_params:
                fixed_params["random_state"] = self.classification_tree_params[
                    "random_state"
                ]
            # Add any other explicitly fixed params (non-None, non-default values)
            # that should not be tuned
            for key, value in self.classification_tree_params.items():
                if key == "random_state":
                    continue  # Already handled
                # Only include as fixed if it's explicitly set (not None and not a default)
                # For now, we'll let the search space handle tuning these
                # Users can explicitly set values in classification_tree_params if they want them fixed

            ct_params, ct_accuracy = self._tune_classification_tree(
                X,
                self.region_types_,
                fixed=fixed_params,  # Only truly fixed params (like random_state)
                search_space=classification_tree_search_space,
                n_trials=classification_tree_tune_n_trials,
                n_splits=classification_tree_tune_n_splits,
                random_state=self.classification_tree_params.get("random_state"),
            )
            print(f"  Best classification tree accuracy: {ct_accuracy:.6f}")
            # Update classification_tree_params with tuned values
            self.classification_tree_params.update(ct_params)
        else:
            print("Training classification tree with provided parameters...")

        self.classification_tree_ = DecisionTreeClassifier(
            **self.classification_tree_params
        )
        self.classification_tree_.fit(X, self.region_types_)

        print("Two-step divergence tree fitting complete!")
        return self

    def _categorize_region_types(
        self, tauF: np.ndarray, tauC: np.ndarray
    ) -> np.ndarray:
        """
        Categorize observations into 4 region types based on treatment effect signs.

        Parameters
        ----------
        tauF : np.ndarray
            Treatment effects for firm outcome.
        tauC : np.ndarray
            Treatment effects for consumer outcome.

        Returns
        -------
        np.ndarray
            Region type labels (1-4).
        """
        region_types = np.zeros(len(tauF), dtype=int)

        # Handle NaN values by treating them as 0
        tauF_clean = np.nan_to_num(tauF, nan=0.0)
        tauC_clean = np.nan_to_num(tauC, nan=0.0)

        # Region 1: both positive
        mask1 = (tauF_clean > 0) & (tauC_clean > 0)
        region_types[mask1] = 1

        # Region 2: firm positive, customer negative
        mask2 = (tauF_clean > 0) & (tauC_clean <= 0)
        region_types[mask2] = 2

        # Region 3: firm negative, customer positive
        mask3 = (tauF_clean <= 0) & (tauC_clean > 0)
        region_types[mask3] = 3

        # Region 4: both negative
        mask4 = (tauF_clean <= 0) & (tauC_clean <= 0)
        region_types[mask4] = 4

        return region_types

    def predict_region_type(self, X: np.ndarray) -> np.ndarray:
        """
        Predict region types for new observations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted region type labels (1-4).
        """
        if self.classification_tree_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        X = np.asarray(X)
        return self.classification_tree_.predict(X)

    def predict_treatment_effects(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict treatment effects for new observations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        tauF : np.ndarray of shape (n_samples,)
            Predicted treatment effects for firm outcome.
        tauC : np.ndarray of shape (n_samples,)
            Predicted treatment effects for consumer outcome.
        """
        if self.causal_forest_F_ is None or self.causal_forest_C_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        X = np.asarray(X)
        tauF = self.causal_forest_F_.effect(X)
        tauC = self.causal_forest_C_.effect(X)

        return tauF, tauC

    def leaf_effects(self) -> Dict[str, Any]:
        """
        Return summary of leaf effects from the classification tree.

        For each leaf in the classification tree, compute average treatment
        effects for observations in that leaf.

        Returns
        -------
        dict
            Dictionary with 'leaves' key containing list of leaf dictionaries.
            Each leaf dict has: leaf_id, region_type, tauF, tauC, n.
        """
        if self.classification_tree_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Get leaf assignments for training data
        leaf_ids = self.classification_tree_.apply(self._fit_data["X"])
        unique_leaves = np.unique(leaf_ids)

        leaves = []
        for leaf_id in unique_leaves:
            mask = leaf_ids == leaf_id
            n = mask.sum()

            # Get average treatment effects for this leaf
            tauF_leaf = np.mean(self.tauF_[mask]) if n > 0 else 0.0
            tauC_leaf = np.mean(self.tauC_[mask]) if n > 0 else 0.0

            # Get most common region type in this leaf
            region_type_leaf = (
                np.bincount(self.region_types_[mask]).argmax() if n > 0 else 0
            )

            leaves.append(
                {
                    "leaf_id": int(leaf_id),
                    "region_type": int(region_type_leaf),
                    "tauF": float(tauF_leaf),
                    "tauC": float(tauC_leaf),
                    "n": int(n),
                }
            )

        return {"leaves": leaves}

    # ===============================================================
    # Classification Tree Tuning Methods
    # ===============================================================

    def _region_type_cv_accuracy(
        self,
        X: np.ndarray,
        region_types: np.ndarray,
        params: Dict[str, Any],
        n_splits: int = 5,
        random_state: Optional[int] = 123,
    ) -> float:
        """
        Compute K-fold cross-validated region type classification accuracy.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        region_types : np.ndarray
            Region type labels (1-4).
        params : dict
            Hyperparameters for DecisionTreeClassifier.
        n_splits : int, default=5
            Number of folds for cross-validation.
        random_state : int, optional
            Random seed for KFold shuffling.

        Returns
        -------
        float
            Mean cross-validated classification accuracy across all folds.
        """
        # Input validation
        n = X.shape[0]
        if len(region_types) != n:
            raise ValueError(
                f"Input arrays must have matching lengths: X={n}, region_types={len(region_types)}"
            )

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        accuracies = []

        for train_idx, val_idx in kf.split(X):
            try:
                clf = DecisionTreeClassifier(**params)
                clf.fit(X[train_idx], region_types[train_idx])
                pred = clf.predict(X[val_idx])
                acc = accuracy_score(region_types[val_idx], pred)
                accuracies.append(acc)
            except Exception:
                accuracies.append(0.0)

        return float(np.mean(accuracies)) if accuracies else 0.0

    def _tune_classification_tree(
        self,
        X: np.ndarray,
        region_types: np.ndarray,
        fixed: Optional[Dict[str, Any]] = None,
        search_space: Optional[Dict[str, Dict[str, Any]]] = None,
        n_trials: int = 30,
        n_splits: int = 5,
        random_state: Optional[int] = 123,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune hyperparameters for classification tree using Optuna.

        Uses Optuna's TPE sampler to optimize classification accuracy, similar
        to the regular divergence tree tuning approach.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        region_types : np.ndarray
            Region type labels (1-4).
        fixed : dict, optional
            Fixed hyperparameters that will be used in all trials.
        search_space : dict, optional
            Search space for classification tree parameters.
            If None, uses default search space.
            Format: {"param_name": {"low": value, "high": value, "log": bool, "step": int}}
        n_trials : int, default=30
            Number of Optuna optimization trials.
        n_splits : int, default=5
            Number of folds for cross-validation.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        best_params : dict
            Best hyperparameters found (combines fixed and tuned parameters).
        best_accuracy : float
            Best cross-validated classification accuracy.
        """
        fixed = dict(fixed or {})
        # Handle search_space: if None, use empty dict (will get defaults)
        # If provided as empty dict {}, respect that (no defaults)
        if search_space is None:
            search_space = {}
        else:
            search_space = dict(search_space)

        # Default search space if not provided and not in fixed
        # Only add defaults if the parameter is not explicitly in fixed
        if "max_depth" not in search_space and "max_depth" not in fixed:
            search_space["max_depth"] = {"low": 2, "high": 15}
        if "min_samples_split" not in search_space and "min_samples_split" not in fixed:
            search_space["min_samples_split"] = {"low": 2, "high": 20}
        if "min_samples_leaf" not in search_space and "min_samples_leaf" not in fixed:
            search_space["min_samples_leaf"] = {"low": 1, "high": 10}

        # Debug: print what will be tuned
        if len(search_space) == 0:
            raise ValueError(
                "No parameters to tune! Search space is empty. "
                "Either provide a search_space or ensure parameters are not all in 'fixed'."
            )

        def objective(trial):
            params = dict(fixed)

            # Suggest hyperparameters from search space
            for name, spec in search_space.items():
                if name in fixed:
                    continue  # Skip if already in fixed params

                if spec.get("log", False):
                    params[name] = trial.suggest_int(
                        name, spec["low"], spec["high"], log=True
                    )
                else:
                    step = spec.get("step", 1)
                    params[name] = trial.suggest_int(
                        name, spec["low"], spec["high"], step=step
                    )

            # Add random_state if provided
            if random_state is not None:
                params["random_state"] = random_state

            accuracy = self._region_type_cv_accuracy(
                X, region_types, params, n_splits=n_splits, random_state=random_state
            )
            return accuracy if np.isfinite(accuracy) else 0.0

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        # Note: Optuna TPE sampler doesn't support parallelization natively
        # For parallel Optuna trials, consider using optuna.study.Study.optimize with n_jobs
        # However, TPE requires sequential trials, so we keep show_progress_bar=False
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        if len(study.trials) == 0 or study.best_trial is None:
            raise RuntimeError(
                "No successful trials completed for classification tree optimization."
            )

        best_params = dict(fixed)
        best_params.update(study.best_trial.params)
        if random_state is not None:
            best_params["random_state"] = random_state

        return best_params, study.best_value
