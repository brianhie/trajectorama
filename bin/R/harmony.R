suppressPackageStartupMessages({
    library(dplyr)
    library(magrittr)
    library(harmony)
})

args = commandArgs(trailingOnly = TRUE)

batch.vector <- scan(file = args[2])

pca.embed <- read.table(args[1], sep = " ")

ptm <- proc.time()

harmony.embed <- HarmonyMatrix(
    pca.embed, batch.vector, "dataset", do_pca = FALSE
)

print(proc.time() - ptm)

write.table(harmony.embed, file = "target/harmony/integrated.txt",
            quote = FALSE, sep = "\t", row.names = FALSE, col.names = FALSE)
