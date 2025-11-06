"""
Divergence Tree for two outcomes using recursive partitioning.

This implementation grows a tree by recursively partitioning the feature space
to identify regions with heterogeneous treatment effects on two outcomes:
- YF: firm outcome (binary; e.g., subscription), observed for all units
- YC: consumer outcome (continuous; observed only when YF == 1, otherwise NaN)

The algorithm:
1. Grows a maximal tree up to max_partitions leaves using global split selection
2. Prunes the tree bottom-up, removing splits with improvement_ratio below threshold

Hyperparameters:
- max_partitions: Maximum number of leaves to grow
- min_improvement_ratio: Minimum improvement ratio required to keep a split

The split selection objective combines heterogeneity and co-movement:
- Heterogeneity: H = zF² + zC²
- Co-movement: d = zF * zC, with φ(d) depending on co_movement mode
- Objective: g = H + λ * φ(d)

where zF and zC are normalized deviations from baseline effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------


@dataclass(eq=False)
class TreeNode:
    """Tree node with split rules and effect estimates."""

    depth: int
    indices: np.ndarray  # Data indices reaching this node
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    # Effect estimates
    tauF: Optional[float] = None
    tauC: Optional[float] = None
    n: int = 0
    n_treated: int = 0
    n_control: int = 0


@dataclass
class PartitionInfo:
    """Information about a partition with its best potential split."""

    leaf: TreeNode
    best_split: Optional[Dict[str, Any]] = None
    best_gain: float = -np.inf


# ---------------------------------------------------------------------
# Main DivergenceTree Class
# ---------------------------------------------------------------------


class DivergenceTree:
    """
    Causal tree for heterogeneous treatment effects on two outcomes.

    This class implements a recursive partitioning algorithm that identifies
    regions with heterogeneous treatment effects on firm (YF) and consumer (YC)
    outcomes. The tree grows maximally to max_partitions leaves, then prunes
    splits with improvement_ratio below min_improvement_ratio.

    Parameters
    ----------
    lambda_ : float, default=1.0
        Weight for co-movement term in objective function.
    max_partitions : int, default=8
        Maximum number of leaves to grow before pruning.
    min_improvement_ratio : float, default=0.01
        Minimum improvement ratio required to keep a split during pruning.
    n_quantiles : int, default=32
        Number of quantiles to consider for continuous feature splits.
    random_state : int, optional
        Random seed for reproducibility (currently unused).
    co_movement : {'both', 'converge', 'diverge'}, default='both'
        Co-movement mode: 'both' (any alignment), 'converge' (both positive),
        'diverge' (opposite signs).
    eps_scale : float, default=1e-8
        Minimum scale value to avoid division by zero.

    Attributes
    ----------
    root_ : TreeNode, optional
        Root node of the fitted tree. None before fit() is called.
    _fit_data : dict
        Internal storage for training data (X, T, YF, YC, indices).
    _root_baseline : dict
        Root-level treatment effects and scales used for objective calculation.
    """

    def __init__(
        self,
        lambda_: float = 1.0,
        max_partitions: int = 8,
        min_improvement_ratio: float = 0.01,
        n_quantiles: int = 32,
        random_state: Optional[int] = None,
        co_movement: str = "both",
        eps_scale: float = 1e-8,
    ):
        self.lambda_ = float(lambda_)
        self.max_partitions = int(max_partitions)
        self.min_improvement_ratio = float(min_improvement_ratio)
        self.n_quantiles = int(n_quantiles)
        self.random_state = random_state
        self.eps_scale = float(eps_scale)

        cm = (co_movement or "both").lower()
        if cm not in {"both", "converge", "diverge"}:
            raise ValueError(
                "co_movement must be one of {'both','converge','diverge'}."
            )
        self.co_movement = cm

        self.root_: Optional[TreeNode] = None
        self._fit_data: Dict[str, Any] = {}
        self._root_baseline: Dict[str, float] = {}

    # ---------------- Public API ----------------

    def fit(
        self, X: np.ndarray, T: np.ndarray, YF: np.ndarray, YC: np.ndarray
    ) -> "DivergenceTree":
        """
        Fit the tree on data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        T : np.ndarray of shape (n_samples,)
            Treatment indicator (0 or 1).
        YF : np.ndarray of shape (n_samples,)
            Firm outcome (binary, observed for all units).
        YC : np.ndarray of shape (n_samples,)
            Consumer outcome (continuous, NaN where YF == 0).

        Returns
        -------
        self : DivergenceTree
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If T contains values other than 0 or 1, or if YC is not NaN
            where YF == 0, or if YC is not finite where YF == 1, or if
            input arrays have mismatched lengths.
        """
        X, T, YF, YC = map(np.asarray, (X, T, YF, YC))

        # Input validation
        n = X.shape[0]
        if len(T) != n or len(YF) != n or len(YC) != n:
            raise ValueError(
                f"Input arrays must have matching lengths: "
                f"X={n}, T={len(T)}, YF={len(YF)}, YC={len(YC)}"
            )

        if not np.all(np.isin(T, [0, 1])):
            raise ValueError("T must be in {0,1}.")
        non_conv_mask = YF == 0
        if not np.all(np.isnan(YC[non_conv_mask])):
            raise ValueError("YC must be NaN where YF == 0 (non-converters).")
        conv_mask = YF == 1
        if not np.all(np.isfinite(YC[conv_mask])):
            raise ValueError("YC must be finite where YF == 1 (converters).")

        all_idx = np.arange(n)

        # Store fit data
        self._fit_data = dict(
            X=X,
            T=T,
            YF=YF,
            YC=YC,
            indices=all_idx,
        )

        # Initialize root node
        self.root_ = TreeNode(depth=0, indices=all_idx)
        tauF, tauC, n_leaf, nt, nc = self._estimate_effects(all_idx)
        self.root_.tauF, self.root_.tauC = tauF, tauC
        self.root_.n, self.root_.n_treated, self.root_.n_control = n_leaf, nt, nc

        # Compute root baseline for 'global' baseline mode
        root_tauF, root_tauC, _, _, _ = self._estimate_effects(all_idx)
        root_sF, root_sC = self._compute_scales(all_idx)
        self._root_baseline = dict(
            tauF=root_tauF,
            tauC=root_tauC,
            sF=root_sF,
            sC=root_sC,
        )

        # Grow maximally to max_partitions
        self._grow()

        # Prune based on improvement_ratio
        self._prune()

        return self

    def predict_leaf(self, X: np.ndarray) -> np.ndarray:
        """
        Return leaf node for each sample in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Array of TreeNode objects, one for each sample's leaf node.
        """
        X = np.asarray(X)

        def descend(x, node: TreeNode) -> TreeNode:
            if node.feature is None or node.left is None or node.right is None:
                return node
            if x[node.feature] <= node.threshold:
                return descend(x, node.left)
            return descend(x, node.right)

        return np.array([descend(x, self.root_) for x in X], dtype=object)

    def leaf_effects(self) -> Dict[str, Any]:
        """
        Return summary of all leaf effects and statistics.

        Returns
        -------
        dict
            Dictionary with 'leaves' key containing list of leaf dictionaries.
            Each leaf dict has: leaf_id, depth, tauF, tauC, n, n_treated, n_control.
        """
        leaves: List[TreeNode] = []

        def collect(n: Optional[TreeNode]):
            if n is None:
                return
            if n.left is None and n.right is None:
                leaves.append(n)
            else:
                collect(n.left)
                collect(n.right)

        collect(self.root_)
        return {
            "leaves": [
                dict(
                    leaf_id=i,
                    depth=leaf.depth,
                    tauF=leaf.tauF,
                    tauC=leaf.tauC,
                    n=leaf.n,
                    n_treated=leaf.n_treated,
                    n_control=leaf.n_control,
                )
                for i, leaf in enumerate(leaves)
            ]
        }

    # ===============================================================
    # Tree Growth (Maximal Tree Construction)
    # ===============================================================

    def _grow(self):
        """Efficient global split selection across all partitions."""
        # Initialize with root partition
        partitions = [PartitionInfo(leaf=self.root_)]

        # Pre-compute best split for root
        self._update_best_split_for_partition(partitions[0])

        while len(partitions) < self.max_partitions:
            # Check if we have any partitions
            if not partitions:
                break

            # Find partition with globally best split
            best_idx = max(
                range(len(partitions)), key=lambda i: partitions[i].best_gain
            )
            best_partition = partitions[best_idx]

            if best_partition.best_gain <= 0:
                break  # No good splits available

            # Apply the best split
            self._apply_split_to_partition(best_partition)

            # Update partitions list
            partitions = self._update_partitions_after_split(partitions, best_idx)

    def _update_best_split_for_partition(self, partition: PartitionInfo):
        """Find and store the best split for a given partition."""
        leaf = partition.leaf

        # Generate all split candidates for this partition
        candidates = self._generate_split_candidates_for_leaf(leaf)

        if not candidates:
            partition.best_split = None
            partition.best_gain = -np.inf
            return

        # Evaluate all candidates and find best
        best_gain = -np.inf
        best_split = None

        for candidate in candidates:
            result = self._evaluate_split_candidate(
                candidate["feature"],
                candidate["threshold"],
                candidate["x_split"],
                leaf.indices,
                leaf,
            )
            if result is not None and result["gain"] > best_gain:
                best_gain = result["gain"]
                best_split = result

        # Store results
        partition.best_split = best_split
        partition.best_gain = best_gain

    def _generate_split_candidates_for_leaf(
        self, leaf: TreeNode
    ) -> List[Dict[str, Any]]:
        """Generate all possible split candidates for a given leaf."""
        X = self._fit_data["X"]
        indices = leaf.indices

        candidates = []
        p = X.shape[1]

        for f in range(p):
            x_split = X[indices, f]
            if np.unique(x_split).size <= 1:
                continue
            qs = np.linspace(0, 1, self.n_quantiles + 2)[1:-1]
            thresholds = np.unique(np.quantile(x_split, qs, method="nearest"))
            for thr in thresholds:
                candidates.append(
                    {
                        "feature": f,
                        "threshold": float(thr),
                        "x_split": x_split,
                    }
                )

        return candidates

    def _apply_split_to_partition(self, partition: PartitionInfo):
        """Apply the best split to a partition."""
        leaf = partition.leaf
        best_split = partition.best_split

        if best_split is None:
            return

        # Apply split to the leaf (improvement_ratio computed during pruning)
        leaf.feature = best_split["feature"]
        leaf.threshold = best_split["threshold"]

        # Create children
        left_mask = best_split["left_mask"]
        right_mask = best_split["right_mask"]
        leaf.left = TreeNode(
            depth=leaf.depth + 1,
            indices=leaf.indices[left_mask],
        )
        leaf.right = TreeNode(
            depth=leaf.depth + 1,
            indices=leaf.indices[right_mask],
        )

        # Compute leaf estimates
        for child, child_indices in [
            (leaf.left, leaf.indices[left_mask]),
            (leaf.right, leaf.indices[right_mask]),
        ]:
            tauF, tauC, n, nt, nc = self._estimate_effects(child_indices)
            child.tauF, child.tauC = tauF, tauC
            child.n, child.n_treated, child.n_control = n, nt, nc

    def _update_partitions_after_split(
        self, partitions: List[PartitionInfo], split_idx: int
    ) -> List[PartitionInfo]:
        """Update partitions after a split, only re-computing affected ones."""
        # Get the split partition
        split_partition = partitions[split_idx]

        # Create new partitions list without the split partition
        new_partitions = partitions[:split_idx] + partitions[split_idx + 1 :]

        # Add the two new child partitions
        left_leaf = split_partition.leaf.left
        right_leaf = split_partition.leaf.right

        left_partition = PartitionInfo(leaf=left_leaf)
        right_partition = PartitionInfo(leaf=right_leaf)

        # Only compute best splits for the two new partitions
        self._update_best_split_for_partition(left_partition)
        self._update_best_split_for_partition(right_partition)

        # Add the two new partitions
        new_partitions.extend([left_partition, right_partition])

        return new_partitions

    def _evaluate_split_candidate(
        self,
        feature: int,
        threshold: float,
        x_split: np.ndarray,
        indices: np.ndarray,
        node: TreeNode,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate single split candidate and return gain if feasible."""
        left_mask = x_split <= threshold
        right_mask = ~left_mask

        # Check minimal feasibility (at least 2 obs per side)
        if left_mask.sum() < 2 or right_mask.sum() < 2:
            return None

        # Compute child effects
        indices_L = indices[left_mask]
        indices_R = indices[right_mask]
        tauF_L, tauC_L, nL, _, _ = self._estimate_effects(indices_L)
        tauF_R, tauC_R, nR, _, _ = self._estimate_effects(indices_R)
        if not all(map(np.isfinite, [tauF_L, tauC_L, tauF_R, tauC_R])):
            return None

        # Get root baseline (always used)
        root_tauF = self._root_baseline.get("tauF", 0.0)
        root_tauC = self._root_baseline.get("tauC", 0.0)
        root_sF = self._root_baseline.get("sF", 1.0)
        root_sC = self._root_baseline.get("sC", 1.0)

        # Calculate weighted objective using root baseline
        gL, compL = self._g(tauF_L, tauC_L, root_tauF, root_tauC, root_sF, root_sC)
        gR, compR = self._g(tauF_R, tauC_R, root_tauF, root_tauC, root_sF, root_sC)
        gain = nL * gL + nR * gR
        if gain <= 0:
            return None

        return dict(
            feature=feature,
            threshold=threshold,
            left_mask=left_mask,
            right_mask=right_mask,
            gain=gain,
            nL=int(nL),
            nR=int(nR),
            components_L=compL,
            components_R=compR,
        )

    # ===============================================================
    # Pruning System
    # ===============================================================

    def _prune(self):
        """
        Prune tree bottom-up, removing splits with improvement_ratio < min_improvement_ratio.

        For each internal node:
        1. Compute total objective with the split (current state)
        2. Compute total objective without the split (if we prune it)
        3. improvement_ratio = (objective_with_split - objective_without_split) / objective_without_split
        4. If improvement_ratio < min_improvement_ratio, prune it
        5. Continue recursively until no more splits can be pruned
        """
        # Keep pruning until no more splits can be removed
        while True:
            # Collect leaf parent nodes (nodes whose children are both leaves)
            leaf_parent_nodes = self._collect_leaf_parent_nodes()

            if not leaf_parent_nodes:
                break  # No more leaf parent nodes to check

            # Find the node with smallest improvement_ratio
            best_node = None
            best_ratio = np.inf

            for node in leaf_parent_nodes:
                improvement_ratio = self._compute_improvement_ratio_for_node(node)
                if improvement_ratio < best_ratio:
                    best_ratio = improvement_ratio
                    best_node = node

            # If the best improvement_ratio is below threshold, prune it
            if best_node is not None and best_ratio < self.min_improvement_ratio:
                self._prune_node_to_leaf(best_node)
            else:
                break  # No more splits to prune

    def _collect_leaf_parent_nodes(self) -> List[TreeNode]:
        """
        Collect nodes whose children are both leaves (leaf parents).

        These are the nodes that directly parent leaves, which are the only
        nodes that need to be checked for pruning, as the node with smallest
        improvement_ratio will always be a leaf parent.
        """
        leaf_parents = []

        def traverse(node: Optional[TreeNode]):
            if node is None:
                return
            if node.left is not None and node.right is not None:
                # Check if both children are leaves
                left_is_leaf = node.left.left is None and node.left.right is None
                right_is_leaf = node.right.left is None and node.right.right is None

                if left_is_leaf and right_is_leaf:
                    # Both children are leaves - this is a leaf parent
                    leaf_parents.append(node)
                else:
                    # Recurse to children to find leaf parents deeper in the tree
                    traverse(node.left)
                    traverse(node.right)

        traverse(self.root_)
        return leaf_parents

    def _compute_improvement_ratio_for_node(self, node: TreeNode) -> float:
        """
        Compute improvement_ratio for a split node.

        improvement_ratio = (objective_with_split - objective_without_split) / objective_without_split

        where:
        - objective_with_split = current total objective (with this split)
        - objective_without_split = total objective if we prune this node (merge children)
        """
        if node.left is None or node.right is None:
            return np.inf  # Not a split node

        # Compute total objective with the split (current state)
        objective_with_split = self._compute_total_objective()

        # Compute total objective without the split (treat this node as a leaf)
        objective_without_split = self._compute_total_objective(prune_node=node)

        if objective_without_split <= 0:
            return np.inf  # Avoid division by zero

        # improvement_ratio = (with_split - without_split) / without_split
        improvement_ratio = (
            objective_with_split - objective_without_split
        ) / objective_without_split

        return improvement_ratio

    def _compute_total_objective(
        self, root: Optional[TreeNode] = None, prune_node: Optional[TreeNode] = None
    ) -> float:
        """
        Compute total objective value across all leaves in the tree.

        Parameters
        ----------
        root : TreeNode, optional
            Root node of the tree to evaluate. If None, uses self.root_
        prune_node : TreeNode, optional
            If provided, treat this node as a leaf (skip its children).
            Used to simulate pruning without copying the tree.

        Returns
        -------
        float
            Total objective value (sum of n * g for all leaves)
        """
        # Use self.root_ if root parameter is not provided
        root = root if root is not None else self.root_

        # Collect all leaves (treating prune_node as a leaf if specified)
        leaves = []

        def collect_leaves(node: Optional[TreeNode], prune_target: Optional[TreeNode]):
            if node is None:
                return
            # If this is the node to prune, treat it as a leaf
            if node is prune_target:
                leaves.append(node)
                return
            if node.left is None and node.right is None:
                leaves.append(node)
            else:
                collect_leaves(node.left, prune_target)
                collect_leaves(node.right, prune_target)

        collect_leaves(root, prune_node)

        # Compute total objective
        total = 0.0
        root_tauF = self._root_baseline.get("tauF", 0.0)
        root_tauC = self._root_baseline.get("tauC", 0.0)
        root_sF = self._root_baseline.get("sF", 1.0)
        root_sC = self._root_baseline.get("sC", 1.0)

        for leaf in leaves:
            # For pruned node, compute merged effect from all data in the node
            if leaf is prune_node:
                tauF, tauC = self._estimate_effects(leaf.indices)[:2]
            else:
                tauF, tauC = leaf.tauF, leaf.tauC

            if tauF is not None and tauC is not None:
                g, _ = self._g(tauF, tauC, root_tauF, root_tauC, root_sF, root_sC)
                total += leaf.n * g

        return total

    def _prune_node_to_leaf(self, node: TreeNode):
        """Prune a node to a leaf by removing its children."""
        # Recompute leaf estimates using all data in this node
        tauF, tauC, n, nt, nc = self._estimate_effects(node.indices)
        node.tauF, node.tauC = tauF, tauC
        node.n, node.n_treated, node.n_control = n, nt, nc

        # Remove split information
        node.feature = None
        node.threshold = None
        node.left = None
        node.right = None

    # ===============================================================
    # Objective Function and Utilities
    # ===============================================================

    def _g(
        self,
        tauF: float,
        tauC: float,
        base_tauF: float,
        base_tauC: float,
        sF: float,
        sC: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute objective function and return components for debugging.

        The objective function combines heterogeneity and co-movement:
        g = zF² + zC² + λ * φ(zF * zC)

        where:
        - zF = (tauF - base_tauF) / sF
        - zC = (tauC - base_tauC) / sC
        - φ depends on co_movement mode

        Returns:
            Tuple of (objective_value, components_dict)
        """
        if not (np.isfinite(tauF) and np.isfinite(tauC)):
            return -np.inf, {
                "zF2": np.nan,
                "zC2": np.nan,
                "phi": np.nan,
                "g": -np.inf,
                "tauF": tauF,
                "tauC": tauC,
                "base_tauF": base_tauF,
                "base_tauC": base_tauC,
                "sF": sF,
                "sC": sC,
            }
        zF = (tauF - base_tauF) / max(sF, self.eps_scale)
        zC = (tauC - base_tauC) / max(sC, self.eps_scale)
        zF2 = float(zF**2)
        zC2 = float(zC**2)
        d = zF * zC
        if self.co_movement == "both":
            phi = float(abs(d))
        elif self.co_movement == "converge":
            phi = float(max(0.0, d))
        else:
            phi = float(max(0.0, -d))
        g = float(zF2 + zC2 + self.lambda_ * phi)
        return g, {
            "zF2": zF2,
            "zC2": zC2,
            "phi": phi,
            "g": g,
            "tauF": tauF,
            "tauC": tauC,
            "base_tauF": base_tauF,
            "base_tauC": base_tauC,
            "sF": sF,
            "sC": sC,
        }

    def _compute_scales(self, idx: np.ndarray) -> Tuple[float, float]:
        """
        Compute standard errors for difference-in-means treatment effect estimates.

        The scales normalize the objective function so firm and consumer effects
        contribute comparably. For tau = mean(Y|T=1) - mean(Y|T=0):
        Var(tau) = Var(Y|T=1)/n1 + Var(Y|T=0)/n0, SE(tau) = sqrt(Var(tau))
        """
        T, YF, YC = (
            self._fit_data["T"],
            self._fit_data["YF"],
            self._fit_data["YC"],
        )
        sub_T, sub_YF, sub_YC = T[idx], YF[idx], YC[idx]
        treated, control = (sub_T == 1), (sub_T == 0)
        n1, n0 = treated.sum(), control.sum()

        # Firm outcome: binary (Bernoulli) variance p(1-p)
        # Var(tauF) = p1(1-p1)/n1 + p0(1-p0)/n0
        p1 = sub_YF[treated].mean() if n1 > 0 else 0.0
        p0 = sub_YF[control].mean() if n0 > 0 else 0.0
        var1 = p1 * (1 - p1) if n1 > 0 else 0.0
        var0 = p0 * (1 - p0) if n0 > 0 else 0.0
        sF2 = max(var1 / max(n1, 1) + var0 / max(n0, 1), self.eps_scale)

        # Consumer outcome: continuous variance (only for converters, YF==1)
        # Var(tauC) = Var(YC|T=1,YF=1)/c1 + Var(YC|T=0,YF=1)/c0
        conv_treat = treated & (sub_YF == 1)
        conv_ctrl = control & (sub_YF == 1)
        c1, c0 = conv_treat.sum(), conv_ctrl.sum()
        varC1 = np.var(sub_YC[conv_treat], ddof=1) if c1 > 1 else 0.0
        varC0 = np.var(sub_YC[conv_ctrl], ddof=1) if c0 > 1 else 0.0
        sC2 = max(varC1 / max(c1, 1) + varC0 / max(c0, 1), self.eps_scale)

        return float(np.sqrt(sF2)), float(np.sqrt(sC2))

    def _estimate_effects(self, idx: np.ndarray) -> Tuple[float, float, int, int, int]:
        """Estimate treatment effects using difference-in-means."""
        T, YF, YC = (
            self._fit_data["T"],
            self._fit_data["YF"],
            self._fit_data["YC"],
        )
        sub_T, sub_YF, sub_YC = T[idx], YF[idx], YC[idx]
        n = idx.size
        treated = sub_T == 1
        control = ~treated
        n1, n0 = treated.sum(), control.sum()

        # Firm effect (binary outcome)
        tauF = np.nan
        if n1 > 0 and n0 > 0:
            tauF = float(sub_YF[treated].mean() - sub_YF[control].mean())

        # Consumer effect (only for converters)
        conv_treat = treated & (sub_YF == 1)
        conv_ctrl = control & (sub_YF == 1)
        c1, c0 = conv_treat.sum(), conv_ctrl.sum()
        tauC = np.nan
        if c1 > 0 and c0 > 0:
            yC1 = sub_YC[conv_treat]
            yC0 = sub_YC[conv_ctrl]
            if np.isfinite(yC1).any() and np.isfinite(yC0).any():
                tauC = float(np.nanmean(yC1) - np.nanmean(yC0))

        return tauF, tauC, int(n), int(n1), int(n0)
