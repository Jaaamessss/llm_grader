from __future__ import annotations

"""
pipeline.py
End-to-end loop:
  Vectorize -> (R or Python) clustering tree -> initial cut
  -> seed editable labels -> [ edit -> inner_product reassign -> promote ]* until user quits

Run:
  python src/pipeline.py
  python src/pipeline.py --fresh --threshold 0.68
  python src/pipeline.py --once
  python src/pipeline.py --no-r

Assumes your repo layout:
  llm_grader/
    artifacts/                 # all outputs live here (single canonical dir)
    data/
    r/protoclust.R
    src/{vectorizer,clustering,inner_product,editor}.py
    config.yaml                # preferred; falls back to src/config.yaml if needed
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

# ---- Local modules ----
try:
    from vectorizer import Vectorizer
    from clustering import Clustering
    from inner_product import InnerProductAssigner
    from editor import ClusterEditor
except Exception as e:
    print("[pipeline] Failed importing local modules:", e)
    print("Make sure you're running from the repo root (python src/pipeline.py) or PYTHONPATH is set.")
    raise


# =========================
# Paths & Config
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]  # .../llm_grader

def find_config() -> Path:
    root_cfg = REPO_ROOT / "config.yaml"
    src_cfg  = REPO_ROOT / "src" / "config.yaml"
    if root_cfg.exists():
        return root_cfg
    if src_cfg.exists():
        return src_cfg
    raise FileNotFoundError(f"config.yaml not found in {root_cfg} or {src_cfg}")

CFG_PATH = find_config()
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

# single canonical artifacts dir at repo root
ARTIFACTS = (REPO_ROOT / CFG.get("artifacts_dir", "artifacts")).resolve()
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# common paths
VECTORS_CSV     = ARTIFACTS / "vectors.csv"
RESPONSES_CSV   = ARTIFACTS / "responses.csv"
PROTO_PC_RDS    = ARTIFACTS / "protoclust_pc.rds"
INITIAL_CSV     = ARTIFACTS / "clusters_initial.csv"
EDITED_CSV      = ARTIFACTS / "clusters_edited.csv"
REFINED_CSV     = ARTIFACTS / "clusters_refined.csv"
CANDIDATES_CSV  = ARTIFACTS / "cluster_candidates.csv"
R_SCRIPT        = REPO_ROOT / "r" / "protoclust.R"


def reconcile_artifact(name: str) -> None:
    """If anything was accidentally written to src/artifacts, move it to the canonical dir."""
    src_alt = REPO_ROOT / "src" / "artifacts" / name
    dst     = ARTIFACTS / name
    if not dst.exists() and src_alt.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        src_alt.replace(dst)

# run once on startup
for fname in [
    "responses.csv", "vectors.csv", "protoclust_pc.rds",
    "clusters_initial.csv", "clusters_edited.csv", "clusters_refined.csv",
    "cluster_candidates.csv",
]:
    reconcile_artifact(fname)


# =========================
# Helpers
# =========================
def ensure_seed_edited(initial_csv: Path, edited_csv: Path) -> None:
    """Create clusters_edited.csv from clusters_initial.csv if it doesn't exist."""
    if not edited_csv.exists():
        if not initial_csv.exists():
            raise FileNotFoundError(f"Cannot seed edits; missing {initial_csv}")
        shutil.copyfile(initial_csv, edited_csv)
        print(f"[pipeline] Seeded {edited_csv.name} from {initial_csv.name}")

def print_round_summary(assign_csv: Path, cand_csv: Path | None) -> None:
    df = pd.read_csv(assign_csv)
    if "cluster" not in df.columns:
        print("[pipeline] WARNING: clusters_refined.csv has no 'cluster' column")
        return
    sizes = df.groupby("cluster")["id" if "id" in df.columns else df.columns[0]].count().sort_index()
    print("\n[pipeline] Current assignments summary (clusters_refined):")
    print(sizes.to_string())
    if cand_csv and Path(cand_csv).exists():
        c = pd.read_csv(cand_csv)
        print(f"\n[pipeline] Candidate re-checks suggested: {len(c)} (saved to {cand_csv})")
        if not c.empty:
            cols = [x for x in [
                "id", "current_cluster", "best_cluster", "second_cluster",
                "max_sim", "second_sim", "margin", "answer"
            ] if x in c.columns]
            print("\nTop 10 candidates (most uncertain first):")
            print(c.sort_values(["max_sim", "margin"], ascending=[True, True])[cols].head(10).to_string(index=False))


# =========================
# Core pipeline
# =========================
def run_pipeline(args: argparse.Namespace) -> None:
    # 1) VECTORIZE
    need_vectorize = args.fresh or (not VECTORS_CSV.exists() or not RESPONSES_CSV.exists())
    if need_vectorize:
        print("\n=== 1) Vectorizing answers ===")
        Vectorizer(str(CFG_PATH)).run()
        # in case modules wrote to src/artifacts
        reconcile_artifact("vectors.csv")
        reconcile_artifact("responses.csv")
    else:
        print("\n=== 1) Vectorizing answers ===\n[skip] vectors already present; use --fresh to recompute")

    # 2) BUILD TREE
    print("\n=== 2) Building clustering tree ===")
    cl = Clustering(str(CFG_PATH))
    built_with_r = False
    if not args.no_r:
        try:
            cl.build()  # R path; expected to write ARTIFACTS/protoclust_pc.rds and plots
            built_with_r = True
        except Exception as e:
            print(f"[pipeline] R protoclust build failed: {e}\n[pipeline] Falling back to Python build.")
    if not built_with_r:
        cl.build_py()  # Python fallback (should write linkage / cached tree under ARTIFACTS)

    # 3) CUT to initial clusters
    print("\n=== 3) Initial cut ===")
    if args.threshold is not None:
        # user provided one explicitly
        if built_with_r and not args.no_r:
            cl.cut(h=args.threshold, out_csv=str(INITIAL_CSV))
        else:
            cl.cut_py(h=args.threshold, out_csv=str(INITIAL_CSV))
    else:
        # fully interactive mode
        cl.run_interactive()

    # keep everything in canonical dir even if module defaulted elsewhere
    for name in ["clusters_initial.csv", "protoclust_pc.rds"]:
        reconcile_artifact(name)

    # Log current cluster sizes
    try:
        init_df = pd.read_csv(INITIAL_CSV)
        if "cluster" in init_df.columns:
            sizes = init_df.groupby("cluster")["id" if "id" in init_df.columns else init_df.columns[0]].count().to_dict()
            print(f"[Clustering] Current number of clusters: {len(sizes)} | sizes: {sizes}")
    except Exception as _:
        pass

    # 4) Seed or validate editable clusters file
    # Always reseed edited from initial
    shutil.copyfile(INITIAL_CSV, EDITED_CSV)
    print(f"[pipeline] Seeded {EDITED_CSV.name} from {INITIAL_CSV.name}")

    # 5) LOOP: edit → assign → promote
    round_idx = 1
    while True:
        print(f"\n=== 4) Editing round {round_idx} ===")
        ce = ClusterEditor(str(EDITED_CSV), str(RESPONSES_CSV))
        # Your editor is assumed to be interactive and to modify its internal df
        # If your editor exposes a different method name, update here.
        if hasattr(ce, "interactive"):
            ce.interactive()
        elif hasattr(ce, "run"):
            ce.run()
        else:
            raise AttributeError("ClusterEditor must expose .interactive() or .run()")

        # Persist edits in-place so the next steps use the updated state
        if hasattr(ce, "save_clusters"):
            ce.save_clusters(str(EDITED_CSV))
        else:
            # Fallback: if editor mutates in memory, write df manually if exposed
            if hasattr(ce, "df"):
                ce.df.to_csv(EDITED_CSV, index=False)
            else:
                raise AttributeError("ClusterEditor must be able to save modified clusters")

        print(f"[pipeline] Saved edits to {EDITED_CSV}")

        # 5b) Inner-product reassignment
        print(f"\n=== 5) Inner-product reassignment (round {round_idx}) ===")
        ip = InnerProductAssigner(
            project_root=REPO_ROOT,
            vectors_csv=str(VECTORS_CSV),
            labels_csv=str(EDITED_CSV),
            responses_csv=str(RESPONSES_CSV),
            out_clusters_csv=str(REFINED_CSV),
            out_candidates_csv=str(CANDIDATES_CSV),
        )
        # Your InnerProductAssigner is assumed to return an object with moved_count & candidates
        res = ip.run(save=True) if "save" in ip.run.__code__.co_varnames else ip.run()
        moved = getattr(res, "moved_count", None)
        if moved is not None:
            print(f"[pipeline] moved={moved}; wrote {REFINED_CSV}")
        else:
            print(f"[pipeline] wrote {REFINED_CSV}")

        candidates = getattr(res, "candidates", None)
        if candidates is not None:
            print(f"[pipeline] wrote candidates -> {CANDIDATES_CSV} (n={len(candidates)})")

        print_round_summary(REFINED_CSV, CANDIDATES_CSV if (CANDIDATES_CSV.exists()) else None)

        # Promote refined -> edited so the NEXT edit starts from the updated state
        shutil.copyfile(REFINED_CSV, EDITED_CSV)
        print(f"[pipeline] Promoted {REFINED_CSV.name} -> {EDITED_CSV.name}")

        if args.once:
            print("[pipeline] --once specified; exiting after one loop.")
            break

        ans = input("\nAnother round of edit + reassignment? [Y/n]: ").strip().lower()
        if ans in {"n", "no"}:
            print("[pipeline] Done. Final clusters in:")
            print(f"  - {EDITED_CSV} (same as clusters_refined.csv)")
            if CANDIDATES_CSV.exists():
                print(f"  - {CANDIDATES_CSV} (candidates/uncertain cases)")
            break

        round_idx += 1


# =========================
# CLI
# =========================
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end clustering/edit/assign pipeline")
    p.add_argument("--reset-edits", action="store_true",
                   help="Overwrite clusters_edited.csv from the current clusters_initial.csv")
    p.add_argument("--fresh", action="store_true", help="Recompute embeddings and rebuild the tree")
    p.add_argument("--threshold", type=float, default=None, help="Initial distance threshold for the first cut")
    p.add_argument("--once", action="store_true", help="Run a single edit→assign loop and exit")
    p.add_argument("--no-r", action="store_true", help="Force Python fallback (skip R/protoclust)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run_pipeline(args)

