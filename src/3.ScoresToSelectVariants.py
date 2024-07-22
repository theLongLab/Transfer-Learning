import sys
import os
import pandas as pd
import numpy as np
import json
import matplotlib
import h5py
import pdb
import configparser
import logging 
from pyliftover import LiftOver

###Definie functions
def scores_hg38_hg19(dis, batch, scores_filenum, logger):
    ###Make a dictionary, for easy search of chr_pos: ref_alt
    sum_polyfun_dict={}
    dis_df = pd.read_parquet("/Enformer_DTL/"+dis+"_dbSNPs_impute_summary_statistic_polyfun_MAF0.01.parquet")
    dis_df.rename(columns={"POSITION":"BP"},inplace=True) #for prostate
    dis_df["CHR"] = dis_df["CHR"].astype("str")
    dis_df["BP"] = dis_df["BP"].astype("str")
    SNP_list=["A","T","C","G"]
    dis_df1 = dis_df[(dis_df["A2"].isin(SNP_list)) & (dis_df["A1"].isin(SNP_list))]
    dis_df_key = list(dis_df1[["CHR","BP"]].agg("_".join, axis=1))
    dis_df_value = list(dis_df1[["A2","A1"]].agg("_".join, axis=1))
    if len(dis_df_key) == len(dis_df_value):
        for i in range(len(dis_df_key)):
            sum_polyfun_dict[dis_df_key[i]]=dis_df_value[i]

    ### Get DTL score files
    ### DTL scores are calcualted based on SNP ID: 1_100000223_C_T, [3] - [2]
    ### GWAS SS beta values are based on A1
    ### If A1 is [3], then fine; else if A1 is [2], DTL scores need to be flipped
    lo=LiftOver('hg38','hg19')  #'hg19', 'hg38'
    root_wk="/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg38_scores/"+batch+"/"
    if not os.path.exists("/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/"+batch+"/"):
        os.system("mkdir -p /Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/"+batch+"/")
    log=open("/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/log.txt","w")

    files=os.listdir(root_wk)
    for file_index in range(49):
        fw=open("/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/"+batch+"/"+str(file_index)+".200K.SNPs.scores.hg19.txt","w")
        with open(root_wk+str(file_index)+".200K.SNPs.scores.txt","r") as fr:
            fr.readline()
            line=fr.readline().strip()
            while line:
                line_list = line.split(",")
                tmp_list = line_list[0].split("_")
                new = lo.convert_coordinate("chr"+tmp_list[0], int(tmp_list[1]))
                if((new is not None) and (len(new)>0)):
                    tmp_list[1] = str(new[0][1])
                    key=tmp_list[0]+"_"+tmp_list[1] #chr_pos
                    if key in sum_polyfun_dict:
                        value = sum_polyfun_dict.get(key)
                        if tmp_list[2]+"_"+tmp_list[3] in value: #REF_ALT
                            line_list[0]="_".join(tmp_list)
                            fw.write(",".join(line_list)+"\n")
                        elif tmp_list[3]+"_"+tmp_list[2] in value: #ALT_REF
                            log.write(line_list[0]+" ref alt have to be flipped\n")
                            snp_values = np.array(line_list[1:]).astype(float)
                            if not np.all(snp_values==0): ##not all zeros
                                snp_values_neg = list((np.negative(snp_values)))
                                snp_scores=[str(snp_value_neg) for snp_value_neg in snp_values_neg]
                                fw.write("_".join(tmp_list)+","+",".join(snp_scores)+"\n")
                        else:
                            log.write(key +" not found in SUMSTATS files\n")
                else:
                    log.write(line_list[0]+" cannot be converted to hg19\n")
                line = fr.readline().strip()
    fw.close()
    log.close()
 
def scores_txtToparquet(dis, batch, scores_filenum, logger):
    hg19_DTL_scores=pd.DataFrame()
    for file_index in range(int(scores_filenum)):
        DTL_df = pd.read_csv("/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/"+batch+"/"+str(file_index)+".200K.SNPs.scores.hg19.txt",sep=",",header=None)
        ori_columns = list(DTL_df.columns)
        new_columns = ["DTL"+str(X) for X in ori_columns]
        DTL_df.columns  = new_columns
        DTL_df["DTL0"] = DTL_df["DTL0"].astype("str")
        DTL_df_refine = DTL_df[DTL_df["DTL0"].str.contains("nan")==False] 
        DTL0_df = pd.DataFrame(list(DTL_df_refine["DTL0"].str.split("_")), columns = ['CHR', 'BP', 'A2', 'A1'])
        DTL_df_refine.index = DTL0_df["CHR"].str.cat([DTL0_df["BP"],DTL0_df["A2"],DTL0_df["A1"]],sep="_")
        DTL_df_ready = DTL_df_refine.drop(columns=['DTL0'])
        DTL_df_ready = DTL_df_ready[~DTL_df_ready.index.duplicated(keep='first')]
        hg19_DTL_scores=pd.concat([hg19_DTL_scores,DTL_df_ready])
    
    for i in range(1,23):
        chr_index=[x for x in list(hg19_DTL_scores.index) if x.startswith(str(i)+"_")]
        logger.info("Number of varnts in "+str(i)+" is "+str(len(chr_index)))
        hg19_DTL_scores_perchr = hg19_DTL_scores.loc[chr_index]
        hg19_DTL_scores_perchr.to_parquet("/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/"+batch+"/"+str(i)+".200K.SNPs.scores.hg19.parquet", engine="pyarrow")
        

### Merge UKB, DTL, Downloaded ENF scores together        
def select_top_variants(dis, batch, top_snps_subset_string):  
    DTL_df_abs_all_1_22=pd.DataFrame()
    for chr_ in range(1,23):
        SUMSTATS=pd.read_parquet("/Enformer_DTL/per_chr/"+dis+"_dbSNPs_impute_summary_statistic_polyfun_MAF0.01_chr"+str(chr_)+".parquet")
        SUMSTATS["CHR"] = SUMSTATS["CHR"].astype("str")
        SUMSTATS["BP"] = SUMSTATS["BP"].astype("str")
        SUMSTATS["A1"] = SUMSTATS["A1"].astype("str")
        SUMSTATS["A2"] = SUMSTATS["A2"].astype("str")
        SUMSTATS_set= set(SUMSTATS["CHR"].str.cat([SUMSTATS["BP"],SUMSTATS["A2"],SUMSTATS["A1"]],sep="_"))
        
        DTL_df_raw=pd.read_parquet("/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/"+batch+"/"+str(chr_)+".200K.SNPs.scores.hg19.parquet")
        DTL_df=DTL_df_raw 
        DTL_df_abs = DTL_df.abs()
        DTL_df_abs["INDEX"] = DTL_df_abs.index
        DTL_df_abs_SUMSTATS = DTL_df_abs[DTL_df_abs["INDEX"].isin(SUMSTATS_set)]
        del DTL_df_abs_SUMSTATS["INDEX"]
        DTL_df_abs_SUMSTATS.to_parquet("/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/"+str(chr_)+".DTL.NumberTracks.parquet")
        DTL_df_abs_all_1_22=pd.concat([DTL_df_abs_all_1_22, DTL_df_abs_SUMSTATS])
    
    ##Select subsets of variants
    gwas_ss=pd.read_csv("/postdoc/Cancer_GWAS_SS/"+dis+"_dbSNPs_impute_summary_statistic_polyfun.txt", sep="\t", engine="pyarrow")
    gwas_ss["CHR"]=gwas_ss["CHR"].astype("str")
    gwas_ss["POSITION"]=gwas_ss["POSITION"].astype("str")
    gwas_ss["varID_hg19"]=gwas_ss[["CHR","POSITION","A2","A1"]].agg("_".join, axis=1)
    tmp = pd.merge(gwas_ss[["SNP","varID_hg19"]], DTL_df_abs_all_1_22, on="varID_hg19", how="inner")

    #Get means for each row
    row_means = tmp.drop(columns=["SNP","varID_hg19","varID"]).mean(axis=1)
    row_mean_df = pd.DataFrame({"SNP":list(tmp["SNP"]), "varID":list(tmp["varID"]),"ABS_MEAN":row_means})
    row_mean_df.reset_index(inplace=True, drop=True)
        
    for snp_set in top_snps_subset_string.split(","):
        #Sort variants by row means and get subset variants
        row_mean_df_sorted = row_mean_df.sort_values(by=['ABS_MEAN'], ascending=False)
        row_mean_df_sorted_subset = row_mean_df_sorted.head(snp_set)
        
        row_mean_df_sorted_subset[["chr","pos","ref","effect"]]=row_mean_df_sorted_subset["varID"].str.split("_",expand=True)
        row_mean_df_sorted_subset["chr"]=row_mean_df_sorted_subset["chr"].astype("int")
        row_mean_df_sorted_subset["pos"]=row_mean_df_sorted_subset["pos"].astype("int")
        row_mean_df_sorted_subset = row_mean_df_sorted_subset.sort_values(by=['chr','pos'])
        row_mean_df_sorted_subset["varID"]="chr"+row_mean_df_sorted_subset["varID"]+"_b38"

        for i in range(1,23):
            row_mean_df_sorted_subset_perchr = row_mean_df_sorted_subset[row_mean_df_sorted_subset["chr"]==i]
            row_mean_df_sorted_subset_perchr.to_csv("/Enformer_DTL/TWAS_2024/subset_used_SNPs/"+model+"_chr"+str(i)+"."+str(int(snp_set//1000))+"K.annot.txt",sep="\t",index=False)
        
def main():
    config=configparser.ConfigParser()
    config.read("3.ScoresToSelectedVariants.ini")

    ###Set up parameters
    dis=config.get("Section1",'dis')
    batch=config.get("Section1",'num_point_tobe_checked')
    scores_filenum=config.get("Section1",'scores_filenum')
    scores_m=config.get("Section1",'scores_methods')  #B, Q

    logging.basicConfig(
        filename=dis+"_"+batch+"_"+scores_m+'.log',  # Name of the log file
        level=logging.DEBUG,     # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
        datefmt='%Y-%m-%d %H:%M:%S'  # Date format
    )
    logger = logging.getLogger('mainLogger')
    
    logger.info('Program started')
    scores_hg38_hg19(dis,batch,scores_filenum, logger)
    print("Finish hg38 to hg19")
    scores_txtToparquet(dis, batch, scores_filenum, logger)
    select_top_variants(dis, batch, top_snps_subset_string)
    logger.info('Program finished')
    
    
if __name__ == '__main__':
    main()
