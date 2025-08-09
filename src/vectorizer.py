# src/vectorizer.py
from __future__ import annotations
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class Vectorizer:
    """
    Reads answers from CSV, encodes with SBERT, L2-normalizes (so dot == cosine),
    and saves artifacts to output_dir.

    Expects config.yaml in the PROJECT ROOT (one level above src/), with keys:
      data_file: "data/combined_answers.csv"
      output_dir: "artifacts"
      sbert_model: "all-MiniLM-L6-v2"
    """
    def __init__(self, config_path: str = "config.yaml"):
        # Always resolve config relative to project root
        project_root = Path(__file__).resolve().parents[1]
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = project_root / cfg_path

        if not cfg_path.exists():
            raise FileNotFoundError(
                f"config.yaml not found at {cfg_path}\n"
                f"Expected it in your project root: {project_root / 'config.yaml'}"
            )

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.data_file = (project_root / self.cfg["data_file"]).resolve()
        self.output_dir = (project_root / self.cfg.get("output_dir", "artifacts")).resolve()
        self.model_name = self.cfg.get("sbert_model", "all-MiniLM-L6-v2")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output paths (under output_dir)
        self.vectors_csv = self.output_dir / "vectors.csv"
        self.embeddings_npy = self.output_dir / "embeddings.npy"
        self.embeddings_meta = self.output_dir / "embeddings_meta.csv"
        self.responses_csv = self.output_dir / "responses.csv"

    def run(self, batch_size: int = 64) -> None:
        # --- load data ---
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        df = pd.read_csv(self.data_file)
        if not {"id", "answer"}.issubset(df.columns):
            raise ValueError("CSV must have columns: id, answer (true_label optional)")

        ids = df["id"].astype(str).tolist()
        texts = df["answer"].astype(str).tolist()

        # --- embed ---
        model = SentenceTransformer(self.model_name)
        embs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit vectors ⇒ cosine == dot
            show_progress_bar=True,
        )

        # --- persist ---
        np.save(self.embeddings_npy, embs)
        pd.DataFrame({"id": ids}).to_csv(self.embeddings_meta, index=False)

        vec_cols = {f"v{i}": embs[:, i] for i in range(embs.shape[1])}
        pd.DataFrame({"id": ids, **vec_cols}).to_csv(self.vectors_csv, index=False)

        keep = [c for c in ("id", "answer", "true_label") if c in df.columns]
        df[keep].to_csv(self.responses_csv, index=False)

        print(
            "[Vectorizer] Saved:\n"
            f"  {self.vectors_csv}\n"
            f"  {self.embeddings_npy}\n"
            f"  {self.embeddings_meta}\n"
            f"  {self.responses_csv}"
        )


if __name__ == "__main__":
    Vectorizer().run()
