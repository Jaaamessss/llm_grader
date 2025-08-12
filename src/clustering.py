# src/clustering.py
from __future__ import annotations
from pathlib import Path
import sys
import subprocess
import yaml
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster


class Clustering:
    """
    Build-once / cut-many clustering with R protoclust (and Python fallback).

    Flow:
      - build()      → compute tree once; write dendrogram/scree; cache tree (RDS or linkage.npy)
      - cut(h)       → fast re-cut at threshold h; writes clusters_initial.csv
      - run_interactive() → guide user through build, then prompt for thresholds repeatedly

    Assumptions:
      - Vectors in artifacts/vectors.csv are L2-normalized so dot == cosine.
      - config.yaml lives in project root (one level above src/).
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Resolve project root and load config
        self.root = Path(__file__).resolve().parents[1]
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = self.root / cfg_path
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # Inputs / outputs
        self.vectors_csv = (self.root / "artifacts" / "vectors.csv").resolve()
        self.out_csv = (self.root / "artifacts" / "clusters_initial.csv").resolve()
        self.dendro_pdf = (self.root / self.cfg.get("dendrogram_file", "artifacts/dendrogram.pdf")).resolve()
        self.scree_pdf = (self.root / self.cfg.get("scree_plot_file", "artifacts/scree_plot.pdf")).resolve()
        self.r_script = (self.root / self.cfg.get("r_script", "r/protoclust.R")).resolve()

        # R cache (tree)
        self.pc_rds = (self.root / "artifacts" / "protoclust_pc.rds").resolve()

        # Python cache (tree)
        self.linkage_npy = (self.root / "artifacts" / "linkage.npy").resolve()
        self.ids_csv = (self.root / "artifacts" / "ids.csv").resolve()

        # Default threshold
        self.distance_threshold = float(self.cfg.get("distance_threshold", 0.68))

    # -----------------------------
    # Utility: print cluster counts
    # -----------------------------
    def _print_cluster_count(self, clusters_csv: Path) -> None:
        try:
            df = pd.read_csv(clusters_csv)
            n = df["cluster"].nunique()
            sizes = df["cluster"].value_counts().sort_index().to_dict()
            print(f"[Clustering] Current number of clusters: {n} | sizes: {sizes}")
        except Exception as e:
            print(f"[Clustering] Could not read clusters file: {e}")

    # ---------------
    # R-mode: BUILD
    # ---------------
    def build(self) -> None:
        if not self.vectors_csv.exists():
            raise FileNotFoundError(f"Vectors not found: {self.vectors_csv}. Run the vectorizer first.")
        cmd = [
            "Rscript",
            str(self.r_script),
            "build",
            str(self.vectors_csv),
            str(self.pc_rds),
            str(self.dendro_pdf),
            str(self.scree_pdf),
        ]
        print("[Clustering:R] build >", " ".join(cmd))
        res = subprocess.run(cmd, cwd=str(self.root), capture_output=True, text=True)
        if res.stdout:
            print(res.stdout.strip())
        if res.returncode != 0:
            if res.stderr:
                print(res.stderr, file=sys.stderr)
            raise RuntimeError("R protoclust build failed")
        print(f"Dendrogram: {self.dendro_pdf}")
        print(f"Scree:      {self.scree_pdf}")

    # -------------
    # R-mode: CUT
    # -------------
    def cut(self, h: float | None = None, out_csv: Path | None = None) -> None:
        h = float(self.distance_threshold if h is None else h)
        out_csv = self.out_csv if out_csv is None else out_csv

        if not self.pc_rds.exists():
            raise FileNotFoundError("No cached R tree; run build() first.")

        cmd = [
            "Rscript",
            str(self.r_script),
            "cut",
            str(self.pc_rds),
            str(h),
            str(out_csv),
        ]
        print("[Clustering:R] cut >", " ".join(cmd))
        res = subprocess.run(cmd, cwd=str(self.root), capture_output=True, text=True)
        if res.stdout:
            print(res.stdout.strip())
        if res.returncode != 0:
            if res.stderr:
                print(res.stderr, file=sys.stderr)
            raise RuntimeError("R protoclust cut failed")

        self._print_cluster_count(out_csv)

    # -------------------
    # Python fallback: BUILD
    # -------------------
    def build_py(self) -> None:
        if not self.vectors_csv.exists():
            raise FileNotFoundError(f"Vectors not found: {self.vectors_csv}. Run the vectorizer first.")
        df = pd.read_csv(self.vectors_csv)
        df["id"] = df["id"].astype(str)
        vec_cols = [c for c in df.columns if c.startswith("v")]
        X = df[vec_cols].to_numpy()

        # Cosine distance on normalized vectors
        D = pdist(X, metric="cosine")
        Z = linkage(D, method="complete")
        np.save(self.linkage_npy, Z)
        df[["id"]].to_csv(self.ids_csv, index=False)
        print(f"[Clustering:Py] built linkage -> {self.linkage_npy}, ids -> {self.ids_csv}")

    # -------------------
    # Python fallback: CUT
    # -------------------
    def cut_py(self, h: float | None = None, out_csv: Path | None = None) -> None:
        h = float(self.distance_threshold if h is None else h)
        out_csv = self.out_csv if out_csv is None else out_csv

        if not self.linkage_npy.exists() or not self.ids_csv.exists():
            raise FileNotFoundError("No cached Python tree; run build_py() first.")

        Z = np.load(self.linkage_npy)
        ids = pd.read_csv(self.ids_csv)["id"].astype(str).tolist()
        cl = fcluster(Z, t=h, criterion="distance")
        out = pd.DataFrame({"id": ids, "cluster": cl})

        # OPTIONAL: compute prototypes in Python fallback using centroids and cosine (dot)
        try:
            vec = pd.read_csv(self.vectors_csv)
            vec["id"] = vec["id"].astype(str)
            vec_cols = [c for c in vec.columns if c.startswith("v")]
            merged = out.merge(vec, on="id", how="left")
            centroids = merged.groupby("cluster")[vec_cols].mean()

            def choose_proto(k: int) -> str:
                g = merged[merged["cluster"] == k]
                c = centroids.loc[k].to_numpy()
                Xk = g[vec_cols].to_numpy()
                sims = Xk @ c
                return str(g.iloc[int(np.argmax(sims))]["id"])

            protos = {int(k): choose_proto(int(k)) for k in centroids.index}
            out["is_prototype"] = out.apply(lambda r: r["id"] == protos[int(r["cluster"])], axis=1)
        except Exception:
            # Fallback: mark all False (you can set prototypes later in review/edit step)
            out["is_prototype"] = False

        out.to_csv(out_csv, index=False)
        print(f"[Clustering:Py] cut h={h:.3f} -> {out_csv}")
        self._print_cluster_count(out_csv)

    # -------------------
    # Interactive flow
    # -------------------
    def run_interactive(self) -> None:
        # Try R first for plots + cached tree
        try:
            self.build()
            while True:
                raw = input(f"Enter new threshold h (current default {self.distance_threshold}) or blank to keep: ").strip()
                h = float(raw) if raw else self.distance_threshold
                self.cut(h)
                again = input("Try another h? [y/N]: ").strip().lower()
                if again != "y":
                    break
        except Exception as e:
            print(f"[Clustering] R failed ({e}); falling back to Python.")
            # Python build/cut loop
            self.build_py()
            while True:
                raw = input(f"(Py) Enter threshold h (default {self.distance_threshold}) or blank: ").strip()
                h = float(raw) if raw else self.distance_threshold
                self.cut_py(h)
                again = input("(Py) Try another h? [y/N]: ").strip().lower()
                if again != "y":
                    break


if __name__ == "__main__":
    Clustering().run_interactive()
