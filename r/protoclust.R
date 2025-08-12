#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(protoclust)
  library(ggplot2)
  suppressWarnings(suppressMessages(library(proxy)))  # for cosine distance
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage:
  Build: Rscript protoclust.R build <vectors_csv> <pc_rds> <dendrogram.pdf> <scree.pdf>
  Cut:   Rscript protoclust.R cut   <pc_rds> <h> <clusters_out.csv>")
}

mode <- args[1]

if (mode == "build") {
  if (length(args) < 5) stop("Build usage: Rscript protoclust.R build <vectors_csv> <pc_rds> <dendrogram.pdf> <scree.pdf>")
  vectors_csv <- args[2]
  pc_rds      <- args[3]
  dendro_pdf  <- args[4]
  scree_pdf   <- args[5]

  df <- read.csv(vectors_csv, check.names = FALSE)
  if (!("id" %in% names(df))) stop("vectors_csv must have an 'id' column")
  ids <- as.character(df$id)

  # Pull vector columns (v0, v1, ...). Adjust pattern if needed.
  vec_cols <- grep("^v[0-9]+$", names(df), value = TRUE)
  if (length(vec_cols) == 0) stop("No vector columns found (expected names like v0, v1, ...)")

  X <- as.matrix(df[, vec_cols, drop = FALSE])
  # Use ids as rownames so they appear in the dendrogram
  rownames(X) <- ids

  # Cosine distance (works directly on your already L2-normalized vectors)
  D <- proxy::dist(X, method = "cosine")

  pc <- protoclust(D)

  # Save both tree and ids so “cut” can write output cleanly
  saveRDS(list(pc = pc, ids = ids), pc_rds)

  # Dendrogram
  pdf(dendro_pdf, width = 10, height = 6)
  plot(pc, main = "Protoclust Dendrogram (cosine)")
  dev.off()

  # Scree-like plot of merge heights
  mh <- pc$height
  scree <- data.frame(step = seq_along(mh), height = mh)
  p <- ggplot(scree, aes(step, height)) +
    geom_line() +
    ggtitle("Merge Heights (Scree)")
  ggsave(filename = scree_pdf, plot = p, width = 8, height = 5)

  cat(sprintf("Built tree -> %s; wrote %s and %s\n", pc_rds, dendro_pdf, scree_pdf))

} else if (mode == "cut") {
  if (length(args) < 4) stop("Cut usage: Rscript protoclust.R cut <pc_rds> <h> <clusters_out.csv>")
  pc_rds      <- args[2]
  h           <- as.numeric(args[3])
  clusters_csv<- args[4]

  obj <- readRDS(pc_rds)
  # Backwards-compatible: allow either list(pc, ids) or plain pc with attr
  if (is.list(obj) && !is.null(obj$pc)) {
    pc  <- obj$pc
    ids <- obj$ids
  } else {
    pc  <- obj
    ids <- attr(pc, "ids")
    if (is.null(ids)) {
      # Final fallback: try using labels from the tree
      ids <- pc$labels
    }
  }
  if (is.null(ids)) stop("Could not recover ids; rebuild with the updated build mode.")

  ct <- protocut(pc, h = h)
  clusters <- ct$cl
  proto_idx <- ct$proto  # NOTE: singular 'proto'

  is_proto <- rep(FALSE, length(ids))
  if (!is.null(proto_idx) && length(proto_idx) > 0) {
    is_proto[proto_idx] <- TRUE
  }

  out <- data.frame(
    id = ids,
    cluster = as.integer(clusters),
    is_prototype = is_proto,
    stringsAsFactors = FALSE
  )
  write.csv(out, clusters_csv, row.names = FALSE)
  cat(sprintf("Cut h=%.3f -> %s\n", h, clusters_csv))

} else {
  stop("First arg must be 'build' or 'cut'")
}
