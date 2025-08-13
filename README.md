# LLM Grader: Interactive Clustering and Reassignment Pipeline

This project implements an **interactive, prototype-based clustering pipeline** for grading or classifying short text responses (e.g., student answers). It uses **SBERT embeddings**, **hierarchical clustering** (with R’s `protoclust` or Python fallback), and an **inner-product reassignment step** to iteratively refine cluster assignments based on user edits.

---

## Features

- **Vectorization** — SBERT embeddings generated from raw answers.
- **Clustering** — R `protoclust` (minimax linkage) with Python fallback.
- **Interactive Editing** — Combine, split, change prototypes, delete clusters.
- **Inner-Product Reassignment** — Reassigns answers to nearest prototypes; flags low-margin “candidates” for review.
- **Iterative Loop** — Edit → Reassign → Edit until satisfied; state is carried forward each round.
- **Configurable** — Paths, model names, and thresholds set in `config.yaml`.

---

## Project Structure
llm_grader/
├── artifacts/ # Generated embeddings, clusters, and candidates
├── config.yaml # Global config (model name, output_dir, etc.)
├── r/
│ └── protoclust.R # R script for minimax linkage clustering
└── src/
├── pipeline.py # Main orchestration script
├── vectorizer.py # SBERT vectorization
├── clustering.py # Clustering build/cut logic
├── editor.py # Interactive cluster editing
└── inner_product.py # Inner-product reassignment logic


> All generated files (CSV, NPY, RDS, PDFs) go into **`artifacts/`** at the repo root.

---

## Requirements

- Python ≥ 3.9
- R with packages:
  - `protoclust`
  - `ggplot2`
- Python packages:
  ```bash
  pip install -r requirements.txt

## Usage
python src/pipeline.py

Inside the interactive editor, you can:
  combine — Merge two clusters into one.
  split — Move selected IDs to a new cluster (auto-creates cluster).
  change_prototype — Set a new prototype for a cluster.
  delete — Remove a cluster entirely.
  show — Display answers and prototypes in a cluster.
  save — Save your changes.
  done — Exit the editor.

## Typical Workflow
Vectorize — SBERT embeddings are created for all answers.
Build Tree — Minimax linkage clustering (R or Python).
Initial Cut — Choose threshold interactively (or pass --threshold).
Edit Round — Adjust clusters interactively.
Reassign — Reassign based on prototypes via cosine similarity.
Repeat — Loop edit → reassign until satisfied.

## Outputs
Generated in artifacts/:
vectors.csv — SBERT embeddings for each answer.
responses.csv — Original answers with IDs.
clusters_initial.csv — Clusters after initial cut.
clusters_edited.csv — Current editable cluster assignments.
clusters_refined.csv — After reassignment step.
cluster_candidates.csv — Ambiguous responses for review.
dendrogram.pdf, scree_plot.pdf
