#!/usr/bin/env Rscript

# conda activate peer
# 2023-10-02
# conda activate peer
# expression matrix must be n x e
# covs_pc_gender_age_num must be n x (5+1+1)
# -------------------------------------------
# SETUP
library(peer)
library(data.table)
library(dplyr)


# print usage
usage <- function() {
  cat(
    'usage: Rscript peer.R tissue_name
    author: Qing Li (liqingbioinfo@gmail.com)
    R version: 3.4+, 3.6.2 recommanded')
}

#Input tissue name
args <- commandArgs(trailingOnly=TRUE)
ts_index <- as.numeric(args[1])
tsList=c("breast","prostate")
ts=tsList[ts_index]

wk="./qli/Enformer_DTL/TWAS_2024/"

#Read in expression matrices
mat <- as.data.frame(fread(paste0(wk,'GTEX_v8_', ts, '_bulk_rnaseq_tpm.qn.invern.csv')))
row_num= dim(mat)[1]
col_num= dim(mat)[2]

mat_gene_expression <- as.matrix(mat[,4:col_num])
class(mat_gene_expression) <- 'numeric'


## ----------------------------------------
#Load genotype PCs, age, gender
#breast_151_covariats.tsv  colontrans_368_covariats.tsv  lung_515_covariats.tsv  prostate_221_covariats.tsv
cov_list=c("breast_151","prostate_221")
cov_prefix=cov_list[ts_index]

covs_age_sex_pc_5_raw = as.data.frame(fread(paste0('./postdoc/MethExpr/GTEx_SC_Bulk/',cov_prefix,'_covariats.tsv'), header=TRUE))
covs_age_sex_pc_5_common=t(covs_age_sex_pc_5_raw)
if(ncol(covs_age_sex_pc_5_common)==ncol(mat_gene_expression)){
  print("Successfully loaded genotype PCs, age and gender")
}else{
  mat_gene_expression <- mat_gene_expression[, colnames(mat_gene_expression) %in% covs_age_sex_pc_5_raw$ID]
  print(ncol(covs_age_sex_pc_5_common)==ncol(mat_gene_expression))
}

# set number of peer factors
## N < 150, use 15  PEERs, 150<=N<250, use 30 PEERs, N >=250 use 35 PEERs
if (ncol(mat_gene_expression) < 150) {
  NumPeerFactorS <- 15
} else if (ncol(mat_gene_expression) < 250) {
  NumPeerFactorS <- 30
} else if (ncol(mat_gene_expression) < 350) {
  NumPeerFactorS <- 45
}else{ # n >= 350
  NumPeerFactorS <- 60
}

print(paste0("Samples in ", ts, "is ", ncol(mat_gene_expression), ". PEER factors according to GTEx should be ", NumPeerFactorS))

#Correct PEER Factors
model <- PEER()
PEER_setPhenoMean(model, t(as.matrix(mat_gene_expression))) # transpose to n x e
PEER_setNk(model,NumPeerFactorS)
PEER_getNk(model)

#Correct gentoype PCs, age, gender
PEER_setCovariates(model, t(as.matrix(covs_age_sex_pc_5_common)))
PEER_update(model)

## ----------------------------------------
# Plots
pdf(paste0(wk, 'GTEX_v8_', ts, '.peer.diag.pdf'), width=6, height=8)
PEER_plotModel(model)
dev.off()

# Outputs
factors = t(PEER_getX(model))
weights = PEER_getW(model)
precision = PEER_getAlpha(model)

residuals = t(PEER_getResiduals(model)) # tranpose back to e x n
colnames(residuals) <- colnames(mat_gene_expression)
residuals_final = cbind(mat[,1:3],residuals)

output_filename=paste0(wk, 'GTEX_v8_', ts, '_bulk_rnaseq_tpm.qn.invern.afterpeer.csv')
fwrite(residuals_final, file=output_filename, row.names = FALSE, col.names = TRUE,sep=",",quote=FALSE)
print(paste0("PEER correction of PEER factors, gentoype PCs, age and gender is finished for ", ts))
