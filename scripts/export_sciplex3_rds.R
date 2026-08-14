#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript export_sciplex3_rds.R INPUT.rds OUTPUT_DIR")
}

input_rds <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(Matrix))

load_optional <- function(package) {
  if (requireNamespace(package, quietly = TRUE)) {
    suppressPackageStartupMessages(library(package, character.only = TRUE))
    return(TRUE)
  }
  FALSE
}

has_monocle3 <- load_optional("monocle3")
has_sce <- load_optional("SingleCellExperiment")
has_se <- load_optional("SummarizedExperiment")
cds <- readRDS(input_rds)

extract_counts <- function(object) {
  candidates <- list(
    function() counts(object),
    function() SummarizedExperiment::assay(object, "counts"),
    function() SummarizedExperiment::assay(object, 1),
    function() object@assayData$exprs,
    function() object@assays@data$counts
  )
  for (candidate in candidates) {
    value <- tryCatch(candidate(), error = function(e) NULL)
    if (!is.null(value)) return(value)
  }
  stop("Unable to locate a counts assay in the RDS object")
}

extract_coldata <- function(object) {
  candidates <- list(
    function() as.data.frame(SummarizedExperiment::colData(object)),
    function() as.data.frame(colData(object)),
    function() as.data.frame(pData(object)),
    function() as.data.frame(object@phenoData@data)
  )
  for (candidate in candidates) {
    value <- tryCatch(candidate(), error = function(e) NULL)
    if (!is.null(value)) return(value)
  }
  stop("Unable to locate cell metadata in the RDS object")
}

extract_rowdata <- function(object) {
  candidates <- list(
    function() as.data.frame(SummarizedExperiment::rowData(object)),
    function() as.data.frame(rowData(object)),
    function() as.data.frame(fData(object)),
    function() as.data.frame(object@featureData@data)
  )
  for (candidate in candidates) {
    value <- tryCatch(candidate(), error = function(e) NULL)
    if (!is.null(value)) return(value)
  }
  data.frame(row.names = rownames(extract_counts(object)))
}

counts_matrix <- extract_counts(cds)
if (!inherits(counts_matrix, "sparseMatrix")) {
  counts_matrix <- Matrix(counts_matrix, sparse = TRUE)
}
cell_metadata <- extract_coldata(cds)
gene_metadata <- extract_rowdata(cds)

if (is.null(rownames(cell_metadata))) rownames(cell_metadata) <- colnames(counts_matrix)
if (is.null(rownames(gene_metadata))) rownames(gene_metadata) <- rownames(counts_matrix)
cell_metadata$cell_id <- rownames(cell_metadata)
gene_metadata$feature_id <- rownames(gene_metadata)
if (!any(c("gene_short_name", "gene_name", "gene_symbol", "symbol") %in% colnames(gene_metadata))) {
  gene_metadata$gene_name <- rownames(gene_metadata)
}

if (ncol(counts_matrix) != nrow(cell_metadata)) {
  stop(sprintf("Counts columns (%d) != cell metadata rows (%d)", ncol(counts_matrix), nrow(cell_metadata)))
}
if (nrow(counts_matrix) != nrow(gene_metadata)) {
  stop(sprintf("Counts rows (%d) != gene metadata rows (%d)", nrow(counts_matrix), nrow(gene_metadata)))
}

matrix_path <- file.path(output_dir, "matrix.mtx")
Matrix::writeMM(counts_matrix, matrix_path)
write.table(cell_metadata, gzfile(file.path(output_dir, "cell_metadata.tsv.gz")), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(gene_metadata, gzfile(file.path(output_dir, "gene_metadata.tsv.gz")), sep = "\t", quote = FALSE, row.names = FALSE)

pigz <- Sys.which("pigz")
if (nzchar(pigz)) {
  status <- system2(pigz, c("-f", "-p", Sys.getenv("SLURM_CPUS_PER_TASK", "4"), matrix_path))
  if (status != 0) stop("pigz failed while compressing matrix.mtx")
}

message("SciPlex3 export complete: ", output_dir)
