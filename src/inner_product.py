from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import yaml


@dataclass
class InnerProductResult:
    assignments: pd.DataFrame          # id, cluster, is_prototype
    candidates: Optional[pd.DataFrame] # may be None if tau/delta disabled
    moved_count: int


def _load_cfg(project_root: Path) -> dict:
    cfg_path = project_root / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class InnerProductAssigner:
    """
    Nearest-prototype assignment via inner product (cosine on unit vectors).

    Reads defaults from config.yaml (override via constructor if desired):
      - inner_product.vectors_csv
      - inner_product.clusters_csv
      - inner_product.responses_csv
      - inner_product.refined_csv
      - inner_product.candidates_csv
      - inner_product.tau
      - inner_product.delta
      - inner_product.ambiguity_across_different_clusters_only
      - inner_product.outlier_cluster
    """

    def __init__(
        self,
        project_root: Optional[str | Path] = None,
        vectors_csv: Optional[str | Path] = None,
        labels_csv: Optional[str | Path] = None,
        responses_csv: Optional[str | Path] = None,
        out_clusters_csv: Optional[str | Path] = None,
        out_candidates_csv: Optional[str | Path] = None,
        tau: Optional[float] = None,
        delta: Optional[float] = None,
        ambiguity_across_diff_only: Optional[bool] = None,
        outlier_cluster: Optional[int] = None,
    ):
        # Resolve project root & load config
        self.root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        cfg = _load_cfg(self.root)
        ip_cfg = cfg.get("inner_product", {}) if isinstance(cfg, dict) else {}
        artifacts_dir = self.root / (cfg.get("output_dir", "artifacts") if isinstance(cfg, dict) else "artifacts")

        # Paths (constructor params override YAML)
        self.vectors_csv = Path(vectors_csv) if vectors_csv else Path(ip_cfg.get("vectors_csv", artifacts_dir / "vectors.csv"))
        self.labels_csv = Path(labels_csv) if labels_csv else Path(ip_cfg.get("clusters_csv", artifacts_dir / "clusters_edited.csv"))
        self.responses_csv = Path(responses_csv) if responses_csv else Path(ip_cfg.get("responses_csv", artifacts_dir / "responses.csv"))
        self.out_clusters_csv = Path(out_clusters_csv) if out_clusters_csv else Path(ip_cfg.get("refined_csv", artifacts_dir / "clusters_refined.csv"))
        self.out_candidates_csv = Path(out_candidates_csv) if out_candidates_csv else Path(ip_cfg.get("candidates_csv", artifacts_dir / "cluster_candidates.csv"))

        # Thresholds / options (constructor overrides YAML)
        self.tau = float(tau) if tau is not None else (float(ip_cfg["tau"]) if "tau" in ip_cfg and ip_cfg["tau"] is not None else None)
        self.delta = float(delta) if delta is not None else (float(ip_cfg["delta"]) if "delta" in ip_cfg and ip_cfg["delta"] is not None else None)
        self.ambiguity_across_diff_only = (
            bool(ambiguity_across_diff_only)
            if ambiguity_across_diff_only is not None
            else bool(ip_cfg.get("ambiguity_across_different_clusters_only", True))
        )
        self.outlier_cluster = outlier_cluster if outlier_cluster is not None else ip_cfg.get("outlier_cluster", None)

    # ---- public API ----
    def run(self, save: bool = True) -> InnerProductResult:
        ids, X = self._load_vectors(self.vectors_csv)
        labels = self._load_labels(self.labels_csv)

        P, P_cids, proto_rows = self._build_prototypes(ids, X, labels)

        # Assign to nearest prototype & keep top-2 sims/indices
        assigned, max_sim, second_sim, t1, t2, S = self._assign_nearest_with_top2(X, P, P_cids)

        # Keep prototypes fixed
        for pr, cl in zip(proto_rows, P_cids):
            assigned[pr] = cl

        # Count moves vs. provided labels (ignore prototypes)
        prev_map = {r["id"]: int(r["cluster"]) for _, r in labels.iterrows()}
        proto_set = set(proto_rows)
        moved = sum(
            1 for i, rid in enumerate(ids)
            if i not in proto_set and prev_map.get(rid) is not None and prev_map[rid] != int(assigned[i])
        )

        # Final assignments
        out = pd.DataFrame({
            "id": ids,
            "cluster": assigned.astype(int),
            "is_prototype": False
        })
        out.loc[list(proto_set), "is_prototype"] = True

        # Optional candidate flagging
        cand_df = None
        if self.tau is not None or self.delta is not None:
            # Best & second-best prototype clusters
            best_cluster = P_cids[t1]
            second_cluster = P_cids[t2]

            # If we only want ambiguity across different clusters, recompute the second-best from a *different* cluster
            if self.ambiguity_across_diff_only and self.delta is not None:
                S_masked = S.copy()
                # remove the best prototype
                S_masked[np.arange(S.shape[0]), t1] = -np.inf
                # mask out all prototypes that belong to the same cluster as the best
                same_cluster_mask = (P_cids[None, :] == best_cluster[:, None])
                S_masked[same_cluster_mask] = -np.inf
                t2_diff = S_masked.argmax(axis=1)
                second_sim = S[np.arange(S.shape[0]), t2_diff]
                second_cluster = P_cids[t2_diff]

            margin = max_sim - second_sim
            is_far = (max_sim < self.tau) if self.tau is not None else np.zeros_like(max_sim, dtype=bool)
            is_amb = (margin < self.delta) if self.delta is not None else np.zeros_like(margin, dtype=bool)
            is_cand = np.logical_or(is_far, is_amb)

            # don't flag prototypes
            mask_proto = np.zeros_like(is_cand, dtype=bool)
            mask_proto[list(proto_set)] = True
            is_cand = np.logical_and(is_cand, ~mask_proto)

            # Build candidates table with current cluster, best & second, and original answer
            current_cluster = labels.set_index("id").loc[ids, "cluster"].to_numpy(int)
            cand_df = pd.DataFrame({
                "id": ids,
                "current_cluster": current_cluster,
                "best_cluster": best_cluster,
                "second_cluster": second_cluster,
                "max_sim": max_sim,
                "second_sim": second_sim,
                "margin": margin,
                "candidate_new_cluster": is_cand
            })
            cand_df = cand_df[cand_df["candidate_new_cluster"]].copy()

            # Attach answer text (if available)
            try:
                resp = pd.read_csv(self.responses_csv).astype({"id": str})
                keep = [c for c in ["id", "answer"] if c in resp.columns]
                if keep:
                    cand_df = cand_df.merge(resp[keep], on="id", how="left")
            except Exception as e:
                print(f"[inner_product] Warning: could not merge responses: {e}")

            # Sort: farthest first, then most ambiguous
            cand_df.sort_values(["max_sim", "margin"], ascending=[True, True], inplace=True)

        if save:
            out.to_csv(self.out_clusters_csv, index=False)
            if cand_df is not None:
                cand_df.to_csv(self.out_candidates_csv, index=False)

        return InnerProductResult(assignments=out, candidates=cand_df, moved_count=moved)

    # ---- helpers ----
    @staticmethod
    def _load_vectors(path: Path) -> Tuple[pd.Series, np.ndarray]:
        df = pd.read_csv(path)
        ids = df["id"].astype(str)
        X = df.drop(columns=["id"]).to_numpy(dtype=np.float32)
        # Safety: normalize (no-op if already unit)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        X = X / norms
        return ids, X

    @staticmethod
    def _load_labels(path: Path) -> pd.DataFrame:
        lab = pd.read_csv(path)
        lab["id"] = lab["id"].astype(str)
        lab["cluster"] = lab["cluster"].astype(int)
        lab["is_prototype"] = lab["is_prototype"].astype(bool)
        return lab

    @staticmethod
    def _build_prototypes(ids: pd.Series, X: np.ndarray, labels: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list[int]]:
        id2row = {i: r for r, i in enumerate(ids)}
        protos = labels[labels["is_prototype"]].sort_values(["cluster", "id"])
        if protos.empty:
            raise ValueError("No prototypes marked in clusters_edited.csv.")
        missing = [pid for pid in protos["id"] if pid not in id2row]
        if missing:
            raise ValueError(f"Prototype ids not in vectors.csv: {missing[:5]}")
        proto_rows = [id2row[pid] for pid in protos["id"]]
        P = X[np.array(proto_rows)]
        P_cids = protos["cluster"].to_numpy(int)
        return P, P_cids, proto_rows

    @staticmethod
    def _assign_nearest_with_top2(X: np.ndarray, P: np.ndarray, P_cids: np.ndarray):
        # Similarity matrix (dot == cosine for unit vectors)
        S = X @ P.T
        # Get top-2 indices per row
        top2 = np.argpartition(-S, 1, axis=1)[:, :2]
        row = np.arange(S.shape[0])[:, None]
        order = np.argsort(-S[row, top2], axis=1)
        top2_sorted = top2[row, order]
        t1 = top2_sorted[:, 0]
        t2 = top2_sorted[:, 1]
        max_sim = S[np.arange(S.shape[0]), t1]
        second_sim = S[np.arange(S.shape[0]), t2]
        assigned = P_cids[t1]
        return assigned, max_sim, second_sim, t1, t2, S


if __name__ == "__main__":
    ip = InnerProductAssigner()
    res = ip.run(save=True)
    sizes = res.assignments.groupby("cluster")["id"].count().sort_index().to_dict()
    print(f"[inner_product] moved={res.moved_count}, clusters={len(sizes)}, sizes={sizes}")
    print(f"[inner_product] saved assignments -> {ip.out_clusters_csv}")
    if res.candidates is not None:
        print(f"[inner_product] saved candidates -> {ip.out_candidates_csv} (n={len(res.candidates)})")
