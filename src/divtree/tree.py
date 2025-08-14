import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from concurrent.futures import as_completed
import multiprocessing as mp


# before:
# @dataclass
# after:
@dataclass(eq=False)
class TreeNode:
    depth: int
    idx_split: np.ndarray  # indices used for split search (honesty: split half)
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    # effects estimated on estimate indices (honesty: estimate half)
    tauF: Optional[float] = None
    tauC: Optional[float] = None
    n: int = 0
    n_treated: int = 0
    n_control: int = 0


class DivergenceTree:
    """
    Recursive partitioning to audit joint heterogeneity in firm-side and consumer-side
    treatment effects.

    Split gain uses a *scaled* joint heterogeneity term plus a co-movement bonus:
       H = ((tauF - parentF)/sF)^2 + ((tauC - parentC)/sC)^2
       d = ((tauF - parentF)/sF) * ((tauC - parentC)/sC)
       phi(d) = |d|      if co_movement == "both"
                max(0,d) if co_movement == "converge"
                max(0,-d) if co_movement == "diverge"
       g = H + lambda_ * phi(d)

    Scales (sF, sC) are the parent-level standard errors of the diff-in-means
    estimators, so each outcome contributes comparably.
    """

    def __init__(
        self,
        lambda_: float = 1.0,
        max_depth: int = 4,
        min_leaf_n: int = 100,
        min_leaf_treated: int = 1,
        min_leaf_control: int = 1,
        min_leaf_conv_treated: int = 1,
        min_leaf_conv_control: int = 1,
        honest: bool = True,
        n_quantiles: int = 32,
        random_state: Optional[int] = None,
        n_workers: int = -1,
        co_movement: str = "both",  # NEW: {"both","diverge","converge"}
        eps_scale: float = 1e-8,  # NEW: numerical floor for scales
    ):
        self.lambda_ = float(lambda_)
        self.max_depth = int(max_depth)
        self.min_leaf_n = int(min_leaf_n)
        self.min_leaf_treated = int(min_leaf_treated)
        self.min_leaf_control = int(min_leaf_control)
        self.min_leaf_conv_treated = int(min_leaf_conv_treated)
        self.min_leaf_conv_control = int(min_leaf_conv_control)
        self.honest = bool(honest)
        self.n_quantiles = int(n_quantiles)
        self.random_state = random_state
        self.n_workers = int(n_workers)

        # --- NEW: co-movement mode + scaling epsilon
        co_movement = (co_movement or "both").lower()
        if co_movement not in {"both", "diverge", "converge"}:
            raise ValueError(
                "co_movement must be one of {'both','diverge','converge'}."
            )
        self.co_movement = co_movement
        self.eps_scale = float(eps_scale)

        self.root_: Optional[TreeNode] = None
        self._fit_data: Dict[str, Any] = {}

    # ---------------- Public API ----------------

    def fit(self, X, T, YF, YC):
        """
        X: features (n,p)
        T: treatment in {0,1}
        YF: firm outcome in {0,1}; YC must be np.nan when YF==0
        YC: consumer outcome (float) for converters only (YF==1)
        """
        X = np.asarray(X)
        T = np.asarray(T)
        YF = np.asarray(YF)
        YC = np.asarray(YC)

        # --- Input checks ---
        if not np.all(np.isin(T, [0, 1])):
            raise ValueError("T must contain only {0,1} values.")
        non_conv_mask = YF == 0
        if not np.all(np.isnan(YC[non_conv_mask])):
            raise ValueError("YC must be np.nan where YF==0 (non-converters).")
        conv_mask = YF == 1
        if not np.all(np.isfinite(YC[conv_mask])):
            raise ValueError("YC must be finite where YF==1 (converters).")

        n, p = X.shape
        rng = np.random.default_rng(self.random_state)
        all_idx = np.arange(n)

        # --- Honesty split ---
        if self.honest:
            mask = rng.integers(0, 2, size=n).astype(bool)
            # ensure both sides non-empty
            if mask.sum() == 0 or (~mask).sum() == 0:
                rng.shuffle(all_idx)
                half = n // 2
                idx_split = all_idx[:half]
                idx_est = all_idx[half:]
            else:
                idx_split = all_idx[mask]
                idx_est = all_idx[~mask]
        else:
            idx_split = all_idx
            idx_est = all_idx

        # --- Store fit data BEFORE any estimation ---
        self._fit_data = {
            "X": X,
            "T": T,
            "YF": YF,
            "YC": YC,
            "idx_split": idx_split,
            "idx_est": idx_est,
        }

        # --- Build root node & estimate root effects (on estimation set) ---
        self.root_ = TreeNode(depth=0, idx_split=idx_split)
        tauF, tauC, n_leaf, nt, nc = self._estimate_effects(idx_est)
        self.root_.tauF, self.root_.tauC = tauF, tauC
        self.root_.n, self.root_.n_treated, self.root_.n_control = n_leaf, nt, nc

        # --- Grow recursively ---
        self._grow(self.root_)
        return self

    def predict_leaf(self, X: np.ndarray):
        """Return the leaf node object for each row in X."""
        X = np.asarray(X)

        def traverse(x, node: TreeNode):
            if node.feature is None or node.left is None or node.right is None:
                return node
            return traverse(
                x, node.left if x[node.feature] <= node.threshold else node.right
            )

        out = []
        for i in range(X.shape[0]):
            out.append(traverse(X[i], self.root_))
        return np.array(out, dtype=object)

    def export_text(self) -> str:
        lines = []

        def rec(node: TreeNode, indent=""):
            if node is None:
                return
            if node.feature is None:
                lines.append(
                    f"{indent}Leaf(depth={node.depth}): "
                    f"tauF={_fmt(node.tauF)}, tauC={_fmt(node.tauC)}, "
                    f"n={node.n}, n1={node.n_treated}, n0={node.n_control}"
                )
                return
            lines.append(f"{indent}X[{node.feature}] <= {node.threshold:.6f}?")
            rec(node.left, indent + "  ")
            lines.append(f"{indent}else:")
            rec(node.right, indent + "  ")

        rec(self.root_)
        return "\n".join(lines)

    def leaf_effects(self) -> Dict[str, Any]:
        leaves = []

        def collect(node: TreeNode):
            if node is None:
                return
            if node.left is None and node.right is None:
                leaves.append(node)
            else:
                collect(node.left)
                collect(node.right)

        collect(self.root_)
        out = []
        for i, leaf in enumerate(leaves):
            out.append(
                dict(
                    leaf_id=i,
                    depth=leaf.depth,
                    tauF=leaf.tauF,
                    tauC=leaf.tauC,
                    n=leaf.n,
                    n_treated=leaf.n_treated,
                    n_control=leaf.n_control,
                )
            )
        return {"leaves": out}

    # --------------- Core tree building ----------------

    def _grow(self, node: TreeNode):
        if node.depth >= self.max_depth:
            return

        X = self._fit_data["X"]
        idx_split = node.idx_split
        idx_est_all = self._fit_data["idx_est"]

        # parent effects on routed estimation indices
        est_idx_parent = idx_est_all[self._route_mask(idx_est_all, upto=node)]
        parent_tauF, parent_tauC, _, _, _ = self._estimate_effects(est_idx_parent)
        if not (np.isfinite(parent_tauF) and np.isfinite(parent_tauC)):
            return

        # --- NEW: compute parent-level scales for normalization ---
        sF, sC = self._compute_scales(est_idx_parent)
        # Guard-rail: if scales are degenerate, stop splitting
        if not (np.isfinite(sF) and np.isfinite(sC)) or (sF <= 0 or sC <= 0):
            return

        p = X.shape[1]

        # Prepare tasks for parallel processing
        tasks = []
        for f in range(p):
            x_split = X[idx_split, f]
            if np.unique(x_split).size <= 1:
                continue

            # candidate thresholds: quantiles over split sample
            qs = np.linspace(0, 1, self.n_quantiles + 2)[1:-1]
            thresholds = np.unique(np.quantile(x_split, qs, method="nearest"))
            if thresholds.size == 0:
                continue

            for thr in thresholds:
                tasks.append((f, thr, x_split))

        # Process tasks in parallel
        best_gain = -np.inf
        best = None

        if tasks:
            from concurrent.futures import ThreadPoolExecutor

            n_workers = (
                mp.cpu_count()
                if self.n_workers == -1
                else min(self.n_workers, len(tasks))
            )
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                for f, thr, x_split in tasks:
                    future = executor.submit(
                        self._evaluate_split_candidate,
                        f,
                        thr,
                        x_split,
                        idx_split,
                        idx_est_all,
                        node,
                        parent_tauF,
                        parent_tauC,
                        sF,  # NEW: pass scales
                        sC,
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None and result["gain"] > best_gain:
                            best_gain = result["gain"]
                            best = result
                    except Exception:
                        continue

        if best is None or best_gain <= 0:
            return

        # apply best split
        f = best["feature"]
        thr = best["threshold"]
        left_node = TreeNode(
            depth=node.depth + 1, idx_split=node.idx_split[best["left_mask_split"]]
        )
        right_node = TreeNode(
            depth=node.depth + 1, idx_split=node.idx_split[best["right_mask_split"]]
        )

        node.feature = f
        node.threshold = thr
        node.left = left_node
        node.right = right_node

        # set child effects (estimated on their estimation indices)
        (
            left_node.tauF,
            left_node.tauC,
            left_node.n,
            left_node.n_treated,
            left_node.n_control,
        ) = self._estimate_effects(best["idx_est_L"])
        (
            right_node.tauF,
            right_node.tauC,
            right_node.n,
            right_node.n_treated,
            right_node.n_control,
        ) = self._estimate_effects(best["idx_est_R"])

        # recurse
        self._grow(left_node)
        self._grow(right_node)

    # --------------- helpers ----------------

    def _evaluate_split_candidate(
        self,
        feature: int,
        threshold: float,
        x_split: np.ndarray,
        idx_split: np.ndarray,
        idx_est_all: np.ndarray,
        node: TreeNode,
        parent_tauF: float,
        parent_tauC: float,
        sF: float,  # NEW
        sC: float,  # NEW
    ) -> Optional[Dict]:
        """Evaluate a single split candidate and return the result if valid."""
        left_mask_split = x_split <= threshold
        right_mask_split = ~left_mask_split

        if (
            left_mask_split.sum() < self.min_leaf_n
            or right_mask_split.sum() < self.min_leaf_n
        ):
            return None

        # route estimation indices with the candidate extra rule
        left_mask_est = self._route_mask(
            idx_est_all, upto=node, extra_rule=(feature, float(threshold), True)
        )
        right_mask_est = self._route_mask(
            idx_est_all, upto=node, extra_rule=(feature, float(threshold), False)
        )
        idx_est_L = idx_est_all[left_mask_est]
        idx_est_R = idx_est_all[right_mask_est]

        # feasibility checks on estimation set
        if not (self._feasible(idx_est_L) and self._feasible(idx_est_R)):
            return None

        tauF_L, tauC_L, nL, _, _ = self._estimate_effects(idx_est_L)
        tauF_R, tauC_R, nR, _, _ = self._estimate_effects(idx_est_R)
        if not (
            np.isfinite(tauF_L)
            and np.isfinite(tauC_L)
            and np.isfinite(tauF_R)
            and np.isfinite(tauC_R)
        ):
            return None

        wL = nL / (nL + nR)
        wR = 1.0 - wL

        # parent g is zero by construction (centered), but keep call for clarity
        g_parent = self._g(parent_tauF, parent_tauC, parent_tauF, parent_tauC, sF, sC)
        g_left = self._g(tauF_L, tauC_L, parent_tauF, parent_tauC, sF, sC)
        g_right = self._g(tauF_R, tauC_R, parent_tauF, parent_tauC, sF, sC)
        gain = wL * g_left + wR * g_right - g_parent

        if gain <= 0:
            return None

        return dict(
            feature=feature,
            threshold=float(threshold),
            left_mask_split=left_mask_split,
            right_mask_split=right_mask_split,
            idx_est_L=idx_est_L,
            idx_est_R=idx_est_R,
            tauF_L=tauF_L,
            tauC_L=tauC_L,
            tauF_R=tauF_R,
            tauC_R=tauC_R,
            gain=gain,
        )

    def _g(
        self,
        tauF: float,
        tauC: float,
        parent_tauF: float,
        parent_tauC: float,
        sF: float,
        sC: float,
    ) -> float:
        """
        Node scoring function.

        Effects are centered and standardized by their parent-node standard errors
        (sF, sC) before computing the heterogeneity and co-movement terms.
        This parent-level scaling ensures both outcomes contribute comparably to
        the split decision regardless of their raw scales, so a single global λ
        remains meaningful throughout the tree.
        """
        if not (
            np.isfinite(tauF)
            and np.isfinite(tauC)
            and np.isfinite(parent_tauF)
            and np.isfinite(parent_tauC)
        ):
            return -np.inf

        # centered & scaled deviations
        zF = (tauF - parent_tauF) / max(sF, self.eps_scale)
        zC = (tauC - parent_tauC) / max(sC, self.eps_scale)

        H = zF**2 + zC**2
        d = zF * zC  # centered cross-moment (covariance-like)

        if self.co_movement == "both":
            phi = abs(d)
        elif self.co_movement == "converge":
            phi = max(0.0, d)
        else:  # "diverge"
            phi = max(0.0, -d)

        return H + self.lambda_ * phi

    def _compute_scales(self, idx: np.ndarray) -> Tuple[float, float]:
        """
        Parent-level scales (standard errors) for diff-in-means:
          sF = sqrt( Var(YF|T=1)/n1 + Var(YF|T=0)/n0 )
          sC = sqrt( Var(YC|T=1,YF=1)/c1 + Var(YC|T=0,YF=1)/c0 )
        These make H and d approximately scale-free across outcomes.
        """
        T = self._fit_data["T"]
        YF = self._fit_data["YF"]
        YC = self._fit_data["YC"]

        sub_T = T[idx]
        sub_YF = YF[idx]
        sub_YC = YC[idx]

        treated = sub_T == 1
        control = ~treated

        n1 = treated.sum()
        n0 = control.sum()

        # Firm-side: Bernoulli variance p*(1-p)
        # Guard against empty arms
        if n1 > 0:
            p1 = float(sub_YF[treated].mean())
            var1 = p1 * (1 - p1)
        else:
            var1 = np.nan
        if n0 > 0:
            p0 = float(sub_YF[control].mean())
            var0 = p0 * (1 - p0)
        else:
            var0 = np.nan

        sF2 = 0.0
        if n1 > 0 and n0 > 0:
            sF2 = var1 / max(n1, 1) + var0 / max(n0, 1)
        sF = float(np.sqrt(max(sF2, self.eps_scale)))

        # Consumer-side: among converters in each arm
        conv_treat = treated & (sub_YF == 1)
        conv_ctrl = control & (sub_YF == 1)
        c1 = conv_treat.sum()
        c0 = conv_ctrl.sum()

        if c1 > 1:
            varC1 = float(np.var(sub_YC[conv_treat], ddof=1))
        elif c1 == 1:
            varC1 = 0.0
        else:
            varC1 = np.nan

        if c0 > 1:
            varC0 = float(np.var(sub_YC[conv_ctrl], ddof=1))
        elif c0 == 1:
            varC0 = 0.0
        else:
            varC0 = np.nan

        sC2 = 0.0
        if c1 > 0 and c0 > 0:
            sC2 = (varC1 / max(c1, 1)) + (varC0 / max(c0, 1))
        sC = float(np.sqrt(max(sC2, self.eps_scale)))

        return sF, sC

    def _estimate_effects(self, idx: np.ndarray) -> Tuple[float, float, int, int, int]:
        T = self._fit_data["T"]
        YF = self._fit_data["YF"]
        YC = self._fit_data["YC"]

        sub_T = T[idx]
        sub_YF = YF[idx]
        sub_YC = YC[idx]

        n = idx.size
        treated = sub_T == 1
        control = ~treated

        n1 = treated.sum()
        n0 = control.sum()

        # firm-side: difference in means
        if n1 == 0 or n0 == 0:
            tauF = np.nan
        else:
            yF1 = sub_YF[treated].mean()
            yF0 = sub_YF[control].mean()
            tauF = yF1 - yF0

        # consumer-side conditional on conversion in each arm
        conv_treat = treated & (sub_YF == 1)
        conv_ctrl = control & (sub_YF == 1)
        c1 = conv_treat.sum()
        c0 = conv_ctrl.sum()
        if c1 == 0 or c0 == 0:
            tauC = np.nan
        else:
            yC1 = sub_YC[conv_treat]
            yC0 = sub_YC[conv_ctrl]
            yC1 = yC1[np.isfinite(yC1)]
            yC0 = yC0[np.isfinite(yC0)]
            if yC1.size == 0 or yC0.size == 0:
                tauC = np.nan
            else:
                tauC = yC1.mean() - yC0.mean()

        return (
            float(tauF) if np.isfinite(tauF) else np.nan,
            float(tauC) if np.isfinite(tauC) else np.nan,
            int(n),
            int(n1),
            int(n0),
        )

    def _feasible(self, idx: np.ndarray) -> bool:
        T = self._fit_data["T"]
        YF = self._fit_data["YF"]
        sub_T = T[idx]
        sub_YF = YF[idx]

        if idx.size < self.min_leaf_n:
            return False

        n1 = (sub_T == 1).sum()
        n0 = idx.size - n1
        if n1 < self.min_leaf_treated or n0 < self.min_leaf_control:
            return False

        c1 = ((sub_T == 1) & (sub_YF == 1)).sum()
        c0 = ((sub_T == 0) & (sub_YF == 1)).sum()
        if c1 < self.min_leaf_conv_treated or c0 < self.min_leaf_conv_control:
            return False

        return True

    def _route_mask(
        self,
        idx_pool: np.ndarray,
        upto: TreeNode,
        extra_rule: Optional[Tuple[int, float, bool]] = None,
    ) -> np.ndarray:
        """
        Build a boolean mask over idx_pool selecting rows that would reach `upto`,
        then optionally apply one more candidate rule (feature f, threshold thr, is_left).
        """
        X = self._fit_data["X"]
        rules = self._path_rules(upto)
        if extra_rule is not None:
            rules = rules + [extra_rule]

        mask = np.ones(idx_pool.size, dtype=bool)
        for f, thr, is_left in rules:
            mask &= (X[idx_pool, f] <= thr) if is_left else (X[idx_pool, f] > thr)
        return mask

    def _path_rules(self, target: TreeNode):
        """
        Reconstruct (feature, threshold, is_left) rules from root to target.
        """
        rules = []

        def dfs(node: Optional[TreeNode], acc):
            if node is None:
                return False
            if node is target:
                rules.extend(acc)
                return True
            if node.feature is None:
                return False
            # try left
            if dfs(node.left, acc + [(node.feature, node.threshold, True)]):
                return True
            # right
            if dfs(node.right, acc + [(node.feature, node.threshold, False)]):
                return True
            return False

        dfs(self.root_, [])
        return rules


def _fmt(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.4f}"
