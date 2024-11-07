library("methods")
library("glmnet")
library("stringr")
library("data.table")
library("dplyr")


get_gene_annotation <- function(gene_annot_file_name, chrom)
{
    gene_df <- read.table(gene_annot_file,header=F,stringsAsFactors =F,sep="\t",fill=T)
    gene_df1 <- filter(gene_df,V3 %in% "gene")
    geneid <- str_extract(gene_df1[,9], "ENSG\\d+.\\d+")
    genename <- gsub("gene_name (\\S+);","\\1",str_extract(gene_df1[,9], "gene_name (\\S+);"), perl=T)
    gene_used <- as.data.frame(cbind(geneid,genename,gene_df1[,c(1,4,5,3)]))
    colnames(gene_used) <- c("geneid","genename","chr","start","end","anno")
    gtf_used <- filter(gene_used,gene_used[,3] %in% (paste0('chr',chrom)))
    gtf_used
}

get_gene_expression <- function(expression_file, gene_annot) {
  expr_df_raw <- as.data.frame(fread(expression_file))
  rownames(expr_df_raw) <-expr_df_raw[,2] #ENSG
  expr_df <- transpose(expr_df_raw[,-c(1:3)])
  colnames(expr_df)<-rownames(expr_df_raw)  #ENSG as columns
  rownames(expr_df)<-colnames(expr_df_raw[,-c(1:3)]) #SampleIDs as rows
  expr_df <- expr_df %>% select(one_of(intersect(gene_annot$geneid, colnames(expr_df)))) #Keep genes presenting in annotations
  as.data.frame(expr_df)
}

##For TEST423
# get_gene_expression <- function(gene_expression_file_name, gene_annot) {
#   expr_df <- as.data.frame(read.table(gene_expression_file_name, header = T, stringsAsFactors = F, row.names = 1))
#   expr_df <- expr_df %>% select(one_of(intersect(gene_annot$geneid, colnames(expr_df))))
#   expr_df <- expr_df[order(row.names(expr_df)), ] # order samples 
#   expr_df
# }

get_maf_filtered_genotype <- function(genotype_file_name, vcfhead, maf, samples) {
  gt_df<- fread(genotype_file,sep="\t",header=F)
  colnames(gt_df) <- vcfhead
  gt_df <- as.data.frame(gt_df)
  row.names(gt_df) <- gt_df$ID
  gt_df <- gt_df[,-c(1:5)]
  gt_df1 <- as.data.frame(t(gt_df))
  gt_df1 <- gt_df1[row.names(gt_df1) %in% samples,]
  #gt_df1 <- gt_df1[,colnames(gt_df1) %in% gwassnp[,2]]
  effect_allele_freqs <- colMeans(gt_df1) / 2
  gt_df1 <- gt_df1[,which((effect_allele_freqs >= maf) & (effect_allele_freqs <= 1-maf))]
  colnames(gt_df1) <- gsub("chr(\\S+)","\\1",colnames(gt_df1),perl=TRUE)
  gt_df1 <- gt_df1[order(row.names(gt_df1)), ]
  gt_df1
}

get_gene_coords <- function(gene_annot, gene) {
  row <- gene_annot[which(gene_annot$geneid == gene),]
  c(row$start, row$end)
}

check_unique <- function(column) {
  length(unique(column)) == 1
}

get_cis_genotype <- function(gt_df, snp_annot, coords, cis_window) {
  snp_info <- snp_annot %>% filter((pos >= (coords[1] - cis_window)  & (pos <= (coords[2] + cis_window))))
  cis_gt <- gt_df[, colnames(gt_df) %in% snp_info$varID]
  cis_gt
}

do_covariance <- function(gene_id, cis_gt, rsids, varIDs, out_file) {
  model_gt <- cis_gt[,varIDs, drop=FALSE]
  geno_cov <- cov(model_gt)
  cov_df <- data.frame(gene=character(),rsid1=character(),rsid2=character(), covariance=double())
  for (i in 1:length(rsids)) {
    for (j in i:length(rsids)) {
      cov_df <- tryCatch(rbind(cov_df, data.frame(gene=gene_id,rsid1=rsids[i], rsid2=rsids[j], covariance=geno_cov[i,j])),
                         error = function(cond) browser())
    }
  }
  write.table(cov_df, file = out_file, append = TRUE, quote = FALSE, col.names = FALSE, row.names = FALSE, sep = " ")
}

check_unique <- function(column) {
  length(unique(column)) == 1
}