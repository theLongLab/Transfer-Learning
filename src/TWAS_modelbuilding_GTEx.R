library("methods")
library("glmnet")
library("stringr")
library("data.table")
library("dplyr")
args <- commandArgs(trailingOnly=TRUE)

###### I. paramter ###### 
tissue="breast"  #breast, prostate
batch="2024"
model="ENF"
wk="./TWAS_2024/"
snpsets=c("200","500","1000","1500","2000","3500")
chrom <- as.character(args[1])
snpset=snpsets[as.numeric(args[2])]
print(snpset)

prefix=paste0(tissue, "_", batch, "_", model, "_", snpset, "K")
print(prefix)

###### II. input files ######
snp_annot_file <- paste0("./TWAS_2024/subset_used_SNPs_hg38/",tissue,"_",batch,"_",model, "_chr", chrom,".", snpset,"K.annot.txt")
expression_file <-  paste0(wk, "GTEX_v8_", tissue, "_bulk_rnaseq_tpm.qn.invern.afterpeer.invern.csv") # qn -> inverse -> peer -> inverse

genotype_file <- paste0("./Genotype_GTEx/genotype_wb/gtexV8_WGS_EUR_blood_genotype_PASS.vcf.chr", chrom, ".gt.gt")
gene_annot_file  <- "/work/long_lab/qli/ref/gencode/gencode.v26.annotation.gene.gtf"

# vcf head
vcfhead <- read.table("./Genotype_GTEx/genotype_wb/vcf.head",header = F, stringsAsFactors = F)

###### III. Set function  ######
source("./GenoPhenoBasic.R")

do_elastic_net <- function(cis_gt, expr_adj, n_folds, cv_fold_ids, n_times, alpha) {
    cis_gt <- as.matrix(cis_gt)
    fit <- cv.glmnet(cis_gt, expr_adj, nfolds = n_folds, alpha = alpha, keep = TRUE, type.measure='mse', foldid = cv_fold_ids[,1], parallel = FALSE)
    lambda_seq <- fit$lambda
    cvms <- matrix(nrow=length(lambda_seq), ncol=n_times)
    fits <- list()
    fits[[1]] <- fit
    cvms <- matrix(nrow = 100, ncol = n_times)
    cvms[1:length(fit$cvm),1] <- fit$cvm
    for (i in 2:(n_times)) {
      fit <- cv.glmnet(cis_gt, expr_adj, lambda = lambda_seq, nfolds = n_folds, alpha = alpha, keep = FALSE, foldid = cv_fold_ids[,i], parallel = FALSE)
      fits[[i]] <- fit
      cvms[1:length(fit$cvm),i] <- fit$cvm
    }
    avg_cvm <- rowMeans(cvms)
    best_lam_ind <- which.min(avg_cvm)
    best_lambda <- lambda_seq[best_lam_ind]
    out <- list(cv_fit = fits[[1]], min_avg_cvm = min(avg_cvm, na.rm = T), best_lam_ind = best_lam_ind, best_lambda = best_lambda)
    out
}

evaluate_performance <- function(cis_gt, expr_adj, fit, best_lam_ind, best_lambda, cv_fold_ids, n_folds) {
  n_nonzero <- fit$nzero[best_lam_ind]
  if (n_nonzero > 0) {
    R2 <- rep(0, n_folds)
    for (j in (1:n_folds)) {
      fold_idxs <- which(cv_fold_ids[,1] == j)
      tss <- sum(expr_adj[fold_idxs]**2)
      rss <- sum((expr_adj[fold_idxs] - fit$fit.preval[fold_idxs, best_lam_ind])**2)
      R2[j] <- 1 - (rss/tss)
    }
    best_fit <- fit$glmnet.fit
    expr_adj_pred <- predict(best_fit, as.matrix(cis_gt), s = best_lambda)
    tss <- sum(expr_adj**2)
    rss <- sum((expr_adj - expr_adj_pred)**2)
    
    n_samp <- length(expr_adj)
    weights <- best_fit$beta[which(best_fit$beta[,best_lam_ind] != 0), best_lam_ind]
    weighted_snps <- names(best_fit$beta[,best_lam_ind])[which(best_fit$beta[,best_lam_ind] != 0)]
    R2_mean <- mean(R2)
    R2_sd <- sd(R2)
    inR2 <- 1 - (rss/tss)
    # Old way
    pred_perf <- summary(lm(expr_adj ~ fit$fit.preval[,best_lam_ind]))
    pred_perf_rsq <- pred_perf$r.squared
    
    pred_perf_pval <- pred_perf$coef[2,4]
    out <- list(weights = weights, n_weights = n_nonzero, weighted_snps = weighted_snps, R2_mean = R2_mean, R2_sd = R2_sd,
                inR2 = inR2, pred_perf_rsq = pred_perf_rsq, pred_perf_pval = pred_perf_pval)
  } else {
    out <- list(weights = NA, n_weights = n_nonzero, weighted_snps = NA, R2_mean = NA, R2_sd = NA,
                inR2 = NA, pred_perf_rsq = NA, pred_perf_pval = NA)
  }
  out
}


###### IV. run analysis ######
# parameter for analysis 
maf=0.05
n_times=3
n_k_folds=10
cis_window=1000000
alpha=0.5

# 1. Inite analysis
gene_annot <- get_gene_annotation(gene_annot_file, chrom)
snp_annot <- fread(snp_annot_file)
#snp_annot$varID <- gsub("chr(\\S+)","\\1",snp_annot$varID,perl=TRUE)

# 2. Expression genes check
expr_df <- get_gene_expression(expression_file, gene_annot)
if(!is.null(expr_df) & (dim(expr_df)[2]!=0)){
  print(paste0("Number of genes to be analyzed ", as.character(dim(expr_df)[2])))
}else{
  print("No genes remain, exit")
  q()
}
genes <- colnames(expr_df)
n_genes <- length(expr_df)

# 3. Organize gentoyep and expression to contain only common subjects
samples<-intersect(vcfhead[6:length(vcfhead)], rownames(expr_df))
n_samples <- length(samples)
print(paste0("Samples for both genotype and phenotype ", as.character(length(samples))))
if(length(samples)<100){
  print("Errors in matching samples in genotype and expression. Please check your gentoype files and expression files!Exit")
  q()}

gt_df <- get_maf_filtered_genotype(genotype_file, vcfhead, maf, rownames(expr_df))
#Order gene expression rows by gt sampels
indices <- match(row.names(gt_df), rownames(expr_df))
expr_df_match_gt <- expr_df[indices, , drop = FALSE]
if(dim(gt_df)[1] != dim(expr_df_match_gt)[1]){
  print("Selected genotype and expression do not have the same number of samples! Exit")
  q()}


# 4. Output model res
seed <- 20240528
set.seed(seed)
cv_fold_ids <- matrix(nrow = n_samples, ncol = n_times)
for (j in 1:n_times)
cv_fold_ids[,j] <- sample(1:n_k_folds, n_samples, replace = TRUE)

# output file
model_summary_file <- paste0(prefix,'_chr',chrom,'_model_summaries.txt')
model_summary_cols <- c('gene_id', 'gene_name', 'alpha', 'cv_mse', 'lambda_iteration', 'lambda_min', 'n_snps_in_model',
					  'cv_R2_avg', 'cv_R2_sd', 'in_sample_R2', 'pred_perf_R2', 'pred_perf_pval')
write(model_summary_cols, file = model_summary_file, ncol = 12, sep = '\t')

weights_file <- paste0(prefix,'_chr',chrom,'_weights.txt')
weights_col <- c('gene_id', 'rsid', 'varID', 'ref', 'alt', 'beta')
write(weights_col, file = weights_file, ncol = 6, sep = '\t')

covariance_file <- paste0(prefix,'_chr',chrom,'_covariances.txt')
covariance_col <- c('gene_id', 'rsid1', 'rsid2', 'corvarianceValues')
write(covariance_col, file = covariance_file, ncol = 4, sep = ' ')

for (i in 1:n_genes) {

	cat(i, "/", n_genes, "\n")
	gene <- genes[i]
	gene_name <- as.character(gene_annot$genename[gene_annot$geneid == gene])
	model_summary <- c(gene, gene_name, alpha, NA, NA, NA, 0, NA, NA, NA, NA, NA)
	coords <- get_gene_coords(gene_annot, gene)
	cis_gt <- get_cis_genotype(gt_df, snp_annot, coords, cis_window)
   
	tryCatch({
		if (ncol(cis_gt) >= 2) {
			adj_expression <- expr_df_match_gt[,i]
			elnet_out <- do_elastic_net(cis_gt, adj_expression, n_k_folds, cv_fold_ids, n_times, alpha)
			if (length(elnet_out) > 0) {
				eval <- evaluate_performance(cis_gt, adj_expression, elnet_out$cv_fit, elnet_out$best_lam_ind, elnet_out$best_lambda, cv_fold_ids, n_k_folds)
				model_summary <- c(gene, as.character(gene_name), alpha, elnet_out$min_avg_cvm, elnet_out$best_lam_ind,
								   elnet_out$best_lambda, eval$n_weights, eval$R2_mean, eval$R2_sd, eval$inR2,
								   eval$pred_perf_rsq, eval$pred_perf_pval)
				if (eval$n_weights > 0) {
				  weighted_snps_info <- snp_annot %>% filter(varID %in% eval$weighted_snps) %>% select(SNP, varID, ref , effect)
				  if (nrow(weighted_snps_info) == 0)
					browser()
				  weighted_snps_info$gene <- gene
				  weighted_snps_info <- weighted_snps_info %>% merge(data.frame(weights = eval$weights, varID=eval$weighted_snps), by = 'varID') %>% select(gene, SNP, varID, ref, effect, weights)
				  write.table(weighted_snps_info, file = weights_file, append = TRUE, quote = FALSE, col.names = FALSE, row.names = FALSE, sep = '\t')
				  do_covariance(gene, cis_gt, weighted_snps_info$SNP, weighted_snps_info$varID, covariance_file)
				}
			}
		}
		write(model_summary, file = model_summary_file, append = TRUE, ncol = 12, sep = '\t')
	},error=function(e){})
}
