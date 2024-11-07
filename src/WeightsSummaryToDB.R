#conda activate twasguo
library("sqldf")
library("stringr")
library("data.table")
library("dplyr")
args <- commandArgs(trailingOnly=T)

###### I. paramter ###### 
tissue="breast"  #colontrans, breast, lung, prostate
batch="2023"
model="DTL"

snpsubsets=c("50K","100K","200K","500K","1000K","1500K","2000K","3500K")
snpsubset=snpsubsets[as.numeric(args[1])]
model_prefix <- paste0(tissue, "_", batch, "_",model,"_", snpsubset)

wk=paste0("/work/long_lab/qli/Enformer_DTL/TWAS_2024/",tissue,"_",batch,"_",model,"/")
# model_prefix <- paste0("CRC_EUR_", CT)
# wk="/work/long_lab/qli/postdoc/MethExpr/Expr_CRC_EUR/TFTWAS_RES/perCT/"


######II. Collect all model summary files [ensure one ENSG name]
model_summary <- matrix(integer(0), nrow = 0, ncol = 6) %>% as.data.frame()
names(model_summary) <- c("gene_id","genename","pred.perf.R2","n.snps.in.model","pred.perf.pval","pred_perf_pval.1")

model_weights <- matrix(integer(0), nrow = 0, ncol = 5) %>% as.data.frame()
names(model_weights) <- c("rsid","gene_id","beta","ref","alt") #gene_id rsid    varID   ref     alt     beta

model_covars <- matrix(integer(0), nrow = 0, ncol = 4) %>% as.data.frame()
names(model_covars) <- c('gene_id', 'rsid1', 'rsid2', 'corvarianceValues')

####Collect model summary#####
for(i in 1:22){
	chrom=as.character(i)
	if (file.exists(paste0(wk,tissue,"_",batch,"_",model,"_",snpsubset,"_chr",chrom,"_model_summaries.txt"))) {
		model_summary_perchr=as.data.frame(fread(paste0(wk,tissue,"_",batch,"_",model,"_",snpsubset,"_chr",chrom,"_model_summaries.txt")))
		if(dim(model_summary_perchr)[2]==12){
			model_summary_perchr_remain=model_summary_perchr[, c(1,2,11,7,12,12)] #$F[0]\t$F[1]\t$F[10]\t$F[6]\t$F[11]\t$F[11]
			model_summary = rbind(model_summary, model_summary_perchr_remain)
		}else{
			cat("Error: reading ",chrom," model summary file does not have 12 columns! Exit")
			q()
		}
	} else {
	  cat("Warning: File not exists ",tissue,"_",batch,"_",model,"_",snpsubset,"_chr",chrom,"_model_summaries.txt")
	}
}
##Remove pred.perf.R2 is NA and keep only one ENSG
model_summary <- model_summary[!is.na(model_summary$pred_perf_R2),]
model_summary <- model_summary %>% arrange(pred_perf_R2)
colnames(model_summary) <- c("gene", "genename" ,"pred.perf.R2", "n.snps.in.model", "pred.perf.pval", "pred.perf.qval")
model_summary_unique <- distinct(model_summary, genename, gene, .keep_all=TRUE)
fwrite(model_summary_unique, paste0(wk, model_prefix, "_model_summaries.csv"), row.names = FALSE, col.names = TRUE)

####Collect model weights#####
for(i in 1:22){
	chrom=as.character(i)
	if (file.exists(paste0(wk,tissue,"_",batch,"_",model,"_",snpsubset,"_chr",chrom,"_weights.txt"))) {
		model_weights_perchr=as.data.frame(fread(paste0(wk,tissue,"_",batch,"_",model,"_",snpsubset,"_chr",chrom,"_weights.txt")))
		if(dim(model_weights_perchr)[2]==6){
			model_weights_perchr_remain=model_weights_perchr[, c(2,1,6,4,5)] #"$F[1],$F[0],$F[5],$F[3],$F[4]"
			model_weights = rbind(model_weights, model_weights_perchr_remain)
		}else{
			cat("Error: reading ",chrom," model weights file does not have 6 columns! Exit")
			q()
		}
	}else{
		cat("Warning: File not exists ",tissue,"_",batch,"_",model,"_",snpsubset,"_chr",chrom,"_weights.txt")
	}
}
model_weights <- model_weights[!is.na(model_weights$beta),]
colnames(model_weights) <- c("rsid","gene","weight","ref_allele","eff_allele")
model_weights_unique <- distinct(model_weights, gene, rsid, .keep_all=TRUE)
fwrite(model_weights_unique, paste0(wk, model_prefix, "_model_weights.csv"), row.names = FALSE, col.names = TRUE)

####Collect model covariates#####
for(i in 1:22){
	chrom=as.character(i)
	if (file.exists(paste0(wk,tissue,"_",batch,"_",model,"_",snpsubset,"_chr",chrom,"_covariances.txt"))) {
		model_covars_perchr=as.data.frame(fread(paste0(wk,tissue,"_",batch,"_",model,"_",snpsubset,"_chr",chrom,"_covariances.txt")))
		if(dim(model_covars_perchr)[2]==4){
			model_covars = rbind(model_covars, model_covars_perchr)
		}else{
			cat("Error: reading ",chrom," model covariances file does not have 6 columns! Exit")
			q()
		}
	}else{
		cat("Warning: File not exists ",tissue,"_",batch,"_",model,"_",snpsubset,"_chr",chrom,"_covariances.txt")
	}
}
model_covars <- model_covars[!is.na(model_covars$corvarianceValues),]
colnames(model_covars) <-c("GENE","RSID1","RSID2","VALUE")
model_covars_unique <- distinct(model_covars, GENE, RSID1, RSID2, .keep_all=TRUE)
fwrite(model_covars_unique, paste0(wk, model_prefix, "_cov.txt"), sep=" ", row.names = FALSE, col.names = TRUE)
system(paste("gzip", paste0(wk, model_prefix, "_cov.txt")))

######III. Generate a db file for SPrediXcan
Summary <- model_summary_unique
Weight <- model_weights_unique
dbfile <- paste0(wk, model_prefix, ".db")

db <- dbConnect(SQLite(), dbname= dbfile)
dbWriteTable(conn = db, name = "extra", value = Summary, row.names = FALSE, header = TRUE)
dbWriteTable(conn = db, name = "weights", value = Weight,row.names = FALSE, header = TRUE)

