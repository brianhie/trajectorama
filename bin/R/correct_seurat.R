options(error=traceback)

library(Seurat)
library(loomR)

library(future)
library(future.apply)
plan("multiprocess", workers = 20)
options(future.globals.maxSize = 20000 * 1024^2)

args = commandArgs(trailingOnly=TRUE)

data.list = list()
#data.list[["1"]] <- as.Seurat(connect(
#    filename = "target/tmp/correct_seurat_2.loom", mode = "r"
#))
#data.list[["2"]] <- as.Seurat(connect(
#    filename = "target/tmp/correct_seurat_4.loom", mode = "r"
#))
data.list[[sprintf("ds%d", 2)]] <- ReadH5AD(
    file = sprintf("target/tmp/correct_seurat_%d.h5ad", 2)
)
data.list[[sprintf("ds%d", 4)]] <- ReadH5AD(
    file = sprintf("target/tmp/correct_seurat_%d.h5ad", 4)
)
#for (i in 1:strtoi(args[1])) {
#    data.list[[sprintf("ds%d", i)]] <- ReadH5AD(
#        file = sprintf("target/tmp/correct_seurat_%d.h5ad", i)
#    )
#}

print(data.list[[1]])

data.list <- lapply(X = data.list, FUN = function(x) {
    x <- NormalizeData(x, verbose = FALSE)
    x <- FindVariableFeatures(x, verbose = FALSE)
})

features <- SelectIntegrationFeatures(object.list = data.list)
data.list <- lapply(X = data.list, FUN = function(x) {
    x <- ScaleData(x, features = features, verbose = FALSE)
    x <- RunPCA(x, features = features, verbose = FALSE)
})

anchors <- FindIntegrationAnchors(
    object.list = data.list, dims = 1:30, reduction = "rpca"
)

#integrated <- IntegrateData(anchorset = anchors, dims = 1:30)
