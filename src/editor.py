# cluster_editor.py
from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union
import pandas as pd


@dataclass
class ClusterSummary:
    cluster: int
    size: int
    prototype_id: Optional[str]
    prototype_answer: Optional[str]


class ClusterEditor:
    """
    Class-based cluster editor for your SBERT + protoclust pipeline.

    Inputs:
      - clusters_initial.csv: columns [id, cluster, is_prototype]
      - responses.csv:        columns [id, answer, ...]
    """

    def __init__(self, clusters_csv: str, responses_csv: str) -> None:
        self.clusters_path = clusters_csv
        self.responses_path = responses_csv
        self.df = self._load_and_join(clusters_csv, responses_csv)
        self._coerce_types()
        self._validate_schema()
        # Ensure IDs are strings for consistency
        self.df["id"] = self.df["id"].astype(str).str.strip()

    # ---------- IO ----------
    def _load_and_join(self, clusters_csv: str, responses_csv: str) -> pd.DataFrame:
        c = pd.read_csv(clusters_csv)
        r = pd.read_csv(responses_csv)

        # Normalize columns (accept serial_number as fallback to id)
        if "id" not in c.columns and "serial_number" in c.columns:
            c = c.rename(columns={"serial_number": "id"})
        if "id" not in r.columns and "serial_number" in r.columns:
            r = r.rename(columns={"serial_number": "id"})

        # Keep only needed columns from responses
        keep_cols = [col for col in ["id", "answer", "true_label"] if col in r.columns]
        r = r[keep_cols].copy()

        # Join
        df = c.merge(r, on="id", how="left", validate="one_to_one")
        return df

    def _coerce_types(self) -> None:
        # cluster as int, is_prototype as bool
        self.df["cluster"] = self.df["cluster"].astype(int)
        if self.df["is_prototype"].dtype == object:
            self.df["is_prototype"] = (
                self.df["is_prototype"].astype(str).str.lower().isin(["true", "1", "yes"])
            )
        else:
            self.df["is_prototype"] = self.df["is_prototype"].astype(bool)

    def _validate_schema(self) -> None:
        required = {"id", "cluster", "is_prototype"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in joined data: {missing}")

        if self.df["id"].duplicated().any():
            dups = self.df[self.df["id"].duplicated()]["id"].tolist()
            raise ValueError(f"Duplicate ids found: {dups[:5]}{'...' if len(dups) > 5 else ''}")

    # ---------- Views ----------
    def summary(self) -> List[ClusterSummary]:
        res: List[ClusterSummary] = []
        for cl, group in self.df.groupby("cluster", sort=True):
            proto_rows = group[group["is_prototype"]]
            pid = proto_rows["id"].iloc[0] if not proto_rows.empty else None
            panswer = proto_rows["answer"].iloc[0] if ("answer" in proto_rows.columns and not proto_rows.empty) else None
            res.append(ClusterSummary(cluster=int(cl), size=len(group), prototype_id=pid, prototype_answer=panswer))
        return res

    def print_summary(self) -> None:
        print("\n=== Cluster Summary ===")
        rows = []
        for s in self.summary():
            rows.append({
                "cluster": s.cluster,
                "size": s.size,
                "prototype_id": s.prototype_id,
                "prototype_answer": (s.prototype_answer[:120] + "…") if s.prototype_answer and len(s.prototype_answer) > 120 else s.prototype_answer
                #"prototype_answer": s.prototype_answer
            })
        print(pd.DataFrame(rows).to_string(index=False))

    def list_cluster(self, cluster_id: int, show_answers: bool = True) -> pd.DataFrame:
        sub = self.df[self.df["cluster"] == int(cluster_id)].copy()
        cols = ["id", "cluster", "is_prototype"]
        if show_answers and "answer" in sub.columns:
            cols.append("answer")
        return sub[cols].sort_values(["is_prototype", "id"], ascending=[False, True])

    # ---------- Edits ----------
    def combine(self, clusters_to_combine: Sequence[int], new_cluster_id: int, new_prototype_id: Optional[str] = None) -> None:
        clusters_to_combine = [int(c) for c in clusters_to_combine]
        self.df.loc[self.df["cluster"].isin(clusters_to_combine), "cluster"] = int(new_cluster_id)

        # If a new prototype is provided, set exactly that one
        if new_prototype_id is not None:
            self._set_single_prototype(int(new_cluster_id), new_prototype_id)
        else:
            # If multiple prototypes now exist in the merged cluster, keep the first and clear the rest
            merged = self.df[self.df["cluster"] == int(new_cluster_id)]
            proto_idx = merged.index[merged["is_prototype"]].tolist()
            if len(proto_idx) > 1:
                # Keep the first encountered
                keep = proto_idx[0]
                self.df.loc[merged.index, "is_prototype"] = False
                self.df.loc[keep, "is_prototype"] = True
            elif len(proto_idx) == 0 and len(merged) > 0:
                # No prototype left—promote the first row as prototype
                self.df.loc[merged.index[0], "is_prototype"] = True

    def split(
            self,
            original_cluster_id: int,
            new_cluster_id: int,
            ids_for_new_cluster: Iterable[Union[str, int]],
            new_prototype_id: Optional[str] = None
    ) -> None:
        # Normalize inputs
        original_cluster_id = int(original_cluster_id)
        new_cluster_id = int(new_cluster_id)
        id_set = {str(x).strip() for x in ids_for_new_cluster}

        # 1) Validate original cluster exists
        clusters_present = set(self.df["cluster"].astype(int).unique())
        if original_cluster_id not in clusters_present:
            raise ValueError(f"Cluster {original_cluster_id} not found. Existing clusters: {sorted(clusters_present)}")

        # 2) Validate IDs belong to the original cluster
        in_orig = self.df[self.df["cluster"] == original_cluster_id]["id"].astype(str)
        missing = sorted(id_set - set(in_orig))
        if missing:
            raise ValueError(
                f"IDs not in cluster {original_cluster_id}: {missing}. "
                f"Tip: run 'show' for {original_cluster_id} to double-check IDs."
            )

        # 3) Move the rows (this implicitly creates the new cluster label)
        move_mask = (self.df["cluster"] == original_cluster_id) & (self.df["id"].astype(str).isin(id_set))
        moved_count = int(move_mask.sum())
        if moved_count == 0:
            raise RuntimeError("Split matched 0 rows to move — check your IDs.")
        self.df.loc[move_mask, "cluster"] = new_cluster_id

        # 4) Prototype handling
        if "is_prototype" not in self.df.columns:
            self.df["is_prototype"] = False

        # Ensure original cluster has exactly one prototype
        self._ensure_cluster_has_one_prototype(original_cluster_id)

        # Ensure new cluster exists and set exactly one prototype there
        new_mask = self.df["cluster"] == new_cluster_id
        if not new_mask.any():
            # Should never happen if moved_count > 0, but guard anyway
            raise RuntimeError(f"Split created no rows for new cluster {new_cluster_id} (unexpected).")

        # Clear any prototype flags in the new cluster
        self.df.loc[new_mask, "is_prototype"] = False

        if new_prototype_id:
            new_prototype_id = str(new_prototype_id).strip()
            if new_prototype_id not in set(self.df.loc[new_mask, "id"].astype(str)):
                raise ValueError(f"Prototype id {new_prototype_id} is not in the new cluster {new_cluster_id}.")
            self.df.loc[new_mask & (self.df["id"].astype(str) == new_prototype_id), "is_prototype"] = True
        else:
            # Default: promote the first moved row as prototype
            first_idx = self.df.index[new_mask][0]
            self.df.loc[first_idx, "is_prototype"] = True


    def change_prototype(self, cluster_id: int, new_prototype_id: Union[str, int]) -> None:
        self._set_single_prototype(int(cluster_id), str(new_prototype_id))

    def delete_cluster(self, cluster_id: int) -> None:
        # Remove all rows in that cluster
        self.df = self.df[self.df["cluster"] != int(cluster_id)].reset_index(drop=True)

    # ---------- Helpers ----------
    def _set_single_prototype(self, cluster_id: int, proto_id: str) -> None:
        proto_id = str(proto_id)
        mask = self.df["cluster"] == int(cluster_id)
        if not mask.any():
            raise ValueError(f"Cluster {cluster_id} not found.")
        if proto_id not in set(self.df.loc[mask, "id"]):
            raise ValueError(f"id {proto_id} not in cluster {cluster_id}.")
        self.df.loc[mask, "is_prototype"] = False
        self.df.loc[mask & (self.df["id"] == proto_id), "is_prototype"] = True

    def _ensure_cluster_has_one_prototype(self, cluster_id: int) -> None:
        mask = self.df["cluster"] == int(cluster_id)
        sub = self.df.loc[mask]
        if sub.empty:
            return
        protos = sub.index[sub["is_prototype"]].tolist()
        if len(protos) == 0:
            # Promote first row to prototype
            self.df.loc[sub.index[0], "is_prototype"] = True
        elif len(protos) > 1:
            # Keep the first, clear others
            keep = protos[0]
            self.df.loc[sub.index, "is_prototype"] = False
            self.df.loc[keep, "is_prototype"] = True

    # ---------- Export ----------
    def save_clusters(self, out_csv: str, include_answer: bool = False) -> None:
        cols = ["id", "cluster", "is_prototype"]
        if include_answer and "answer" in self.df.columns:
            cols += ["answer"]
        self.df[cols].to_csv(out_csv, index=False)

    # ---------- CLI ----------
    def interactive(self) -> None:
        self.print_summary()
        while True:
            cmd = input("\nCommand (combine / split / change_prototype / delete / show / save / done): ").strip().lower()
            try:
                if cmd == "combine":
                    ids_str = input("Clusters to combine (e.g., 3 7): ").strip()
                    new_id = int(input("New cluster ID: ").strip())
                    proto = input("Prototype id for new cluster (blank = auto): ").strip() or None
                    clusters = [int(x) for x in ids_str.split()]
                    self.combine(clusters, new_id, proto)
                    print(f"Combined {clusters} -> {new_id}")

                elif cmd == "split":
                    original = int(input("Original cluster ID: ").strip())
                    new_id = int(input("New cluster ID: ").strip())
                    ids_str = input("IDs to move to new cluster (space-separated): ").strip()
                    ids_new = [x for x in ids_str.split()]
                    proto = input("Prototype id for new cluster (blank = auto): ").strip() or None
                    self.split(original, new_id, ids_new, proto)
                    print(f"Split {original} -> {original} & {new_id}")

                elif cmd == "change_prototype":
                    cl = int(input("Cluster ID: ").strip())
                    pid = input("New prototype id: ").strip()
                    self.change_prototype(cl, pid)
                    print(f"Prototype for cluster {cl} set to {pid}")

                elif cmd == "delete":
                    cl = int(input("Cluster ID to delete: ").strip())
                    self.delete_cluster(cl)
                    print(f"Deleted cluster {cl}")

                elif cmd == "show":
                    cl = int(input("Cluster ID to show: ").strip())
                    print(self.list_cluster(cl).to_string(index=False))

                elif cmd == "save":
                    out = input("Output CSV (default: clusters_edited.csv): ").strip() or "clusters_edited.csv"
                    self.save_clusters(out)
                    print(f"Saved to {out}")

                elif cmd in ("done", "exit", "quit"):
                    break

                else:
                    print("Unknown command.")
            except Exception as e:
                print(f"Error: {e}")
            self.print_summary()


# --------- Script usage ---------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cluster_editor.py <clusters_initial.csv> <responses.csv> [--interactive] [--out clusters_edited.csv]")
        sys.exit(1)

    clusters_csv = sys.argv[1]
    responses_csv = sys.argv[2]
    interactive = "--interactive" in sys.argv
    try:
        out_idx = sys.argv.index("--out")
        out_csv = sys.argv[out_idx + 1]
    except ValueError:
        out_csv = "clusters_edited.csv"
    ce = ClusterEditor(clusters_csv, responses_csv)
    if interactive:
        ce.interactive()
        ce.save_clusters(out_csv)
    else:
        ce.print_summary()
        ce.save_clusters(out_csv)
