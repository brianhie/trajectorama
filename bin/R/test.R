library(SeuratData)
library(Seurat)
library(ggplot2)

# set up future for parallelization
library(future)
library(future.apply)
plan("multiprocess", workers = 4)
options(future.globals.maxSize = 20000 * 1024^2)

# load in the data
data(hcabm40k)
bm40k.list <- SplitObject(hcabm40k, split.by = "orig.ident")
bm40k.list <- lapply(X = bm40k.list, FUN = function(x) {
    x <- NormalizeData(x, verbose = FALSE)
    x <- FindVariableFeatures(x, verbose = FALSE)
})

features <- SelectIntegrationFeatures(object.list = bm40k.list)
bm40k.list <- lapply(X = bm40k.list, FUN = function(x) {
    x <- ScaleData(x, features = features, verbose = FALSE)
    x <- RunPCA(x, features = features, verbose = FALSE)
})
typeof(bm40k.list[[1]])

anchors <- FindIntegrationAnchors(object.list = bm40k.list,# reference = c(1, 2),
                                  reduction = "rpca", dims = 1:30)
bm40k.integrated <- IntegrateData(anchorset = anchors, dims = 1:30)
