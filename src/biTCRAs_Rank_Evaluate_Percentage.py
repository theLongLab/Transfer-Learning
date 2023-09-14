import numpy as np
import pandas as pd
import os
import re
import json
from pandas_plink import read_plink1_bin
from pandas_plink import write_plink1_bin
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sys
from pyliftover import LiftOver

##Stop and return the index of a value that nearest larger than target value
def binary_search(a, x):  ##find the last value that is less or equal to the target position value
    lo=-1
    hi=len(a)
    while lo +1 != hi:
        mid = (lo+hi)//2
        midval = a[mid]
        if midval <= x:
            lo = mid
        else: 
            hi = mid
    return lo
        
def chr_pos_strat_end_lists(prefix,category):
    chr_pos_start_lists=[]
    chr_pos_start_lists_sorted=[]
    chr_pos_start_end_dict={}
    for i in range(22):
        chr_pos_start_lists.append(list())

    f=open("/work/long_lab/qli/Breat_TFs_SNPs/Enhancer_Promoters_Annotations/roadmap/"+prefix+"_15_coreMarks_hg38lift_segments_"+category+".bed","r")
    
    header=f.readline()
    line=f.readline().strip()
    while line:
        if not line.startswith("chrX"):
            line_arr = line.split("\t")
            chr_=int(re.search(r'\d{1,2}',line_arr[0]).group())
            chr_pos_start_lists[chr_-1].append(int(line_arr[1]))
            key=str(chr_)+"_"+line_arr[1]
            chr_pos_start_end_dict[key]=key+"_"+line_arr[2]
            line=f.readline().strip()
        else:
            break
    f.close()
    
    for i in range(22):
        chr_pos_start_lists_sorted.append(sorted(chr_pos_start_lists[i]))
        
    return chr_pos_start_lists_sorted,chr_pos_start_end_dict
        
def SNPs_percent(K,N_K):
    return(len(K), len(K)+len(N_K), len(K)/len(K+N_K))
    
def number_SNPs_in_out_of_func_regions(target_SNPs_set,prefix,category):
    flank=0
    SNP_in_func_regions=[]
    SNP_outof_func_regions=[]
    
    chr_pos_start_lists_sorted,chr_pos_start_end_dict = chr_pos_strat_end_lists(prefix,category)
    for snp in target_SNPs_set:
        chr_,pos,ref,alt=snp.split("_")
        if not chr_=="23":
            chr_tmp=chr_pos_start_lists_sorted[int(chr_)-1]
            pos_index = binary_search(chr_tmp,int(pos))
            
            if pos_index == len(chr_tmp)-1:  ###all positions are less than target pos
                front_start = behind_start = chr_tmp[pos_index]
            elif pos_index == -1: ###all posiitons are larger than target pos
                front_start = behind_start = chr_tmp[pos_index+1]
            else:
                behind_start = chr_tmp[pos_index+1]
                front_start = chr_tmp[pos_index]

            front_start_end = chr_pos_start_end_dict.get(chr_+"_"+str(front_start))
            behind_start_end = chr_pos_start_end_dict.get(chr_+"_"+str(behind_start))

            ##check if the pos is located in the front or behind gene
            _,fs,fe=front_start_end.split("_")
            _,bs,be=behind_start_end.split("_")

            if (int(pos)<=int(fe) and int(pos)>=int(fs)) or (int(pos)<=int(be) and int(pos)>=int(bs)):
                SNP_in_func_regions.append(snp)
                print(snp,fs,fe,bs,be)
            else:
                SNP_outof_func_regions.append(snp)  
#         else:
#             print(snp)
    return SNP_in_func_regions,SNP_outof_func_regions

def process(SnpNumsString, model_name, StatesIndex):
    wk="/work/long_lab/qli/Breat_TFs_SNPs/Enhancer_Promoters_Annotations/roadmap/percentage_res/"
    InputFile="/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/breast/1_22."+model_name+".NumberTracks.parquet"
    cell_lines=["E027","E028","E029","E055","E066","E119","E121","E124","E125","E126"]
    
    ###Load functional SNPs
    functional_SNP_df_raw = pd.read_parquet(InputFile)

    ###Load GWAS summary statistic file
    gwas_sumstats = pd.read_csv("/work/long_lab/qli/Breat_TFs_SNPs/breast/breast_GWAS_TF.matrix.hg38.txt",sep="\t")
    gwas_sumstats["MAF"] = list(np.where(gwas_sumstats["EAF"]>0.5, 1-gwas_sumstats["EAF"], gwas_sumstats["EAF"]))
    gwas_sumstats["ID_hg38"]="chr"+gwas_sumstats["CHR_Position_A2_A1"]+"_b38"
    
    ###Identify top SnpNum SNPs
    SnpNumsList=[]
    if ";" in SnpNumsString:
        SnpNumsList.extend(SnpNumsString.split(";"))
    else:
        SnpNumsList.append(SnpNumsString)
    
    for SnpNum in SnpNumsList:
        SNP_tobetested = pd.read_csv("/work/long_lab/qli/Enformer_DTL/SCORES_TWAS/SCORES_ANNOTATIONS/breast_"+model_name+"_"+str(SnpNum)+".txt",header=None)
        functional_SNP_tobetested = gwas_sumstats[gwas_sumstats["ID_hg38"].isin(set(SNP_tobetested.loc[:,0]))]
        
        ###Number of MAF SNPs
        functional_SNP_tobetested1 = functional_SNP_tobetested[functional_SNP_tobetested["MAF"]<0.01].shape[0]
        functional_SNP_tobetested2 = functional_SNP_tobetested[(functional_SNP_tobetested["MAF"]>=0.01) & (functional_SNP_tobetested["MAF"]<0.05)].shape[0]
        functional_SNP_tobetested3 = functional_SNP_tobetested[(functional_SNP_tobetested["MAF"]>=0.05) & (functional_SNP_tobetested["MAF"]<0.1)].shape[0]
        functional_SNP_tobetested4 = functional_SNP_tobetested[(functional_SNP_tobetested["MAF"]>=0.1) & (functional_SNP_tobetested["MAF"]<0.2)].shape[0]
        functional_SNP_tobetested5 = functional_SNP_tobetested[functional_SNP_tobetested["MAF"]>=0.2].shape[0]
        maf_list=[functional_SNP_tobetested1,functional_SNP_tobetested2,functional_SNP_tobetested3,functional_SNP_tobetested4,functional_SNP_tobetested5]
        maf_list_df=pd.DataFrame({"MAF":maf_list})
        
        ###Number of GWAS risk SNPs
        functional_SNP_tobetested["P"]=functional_SNP_tobetested["P"].astype("float")
        is_gwas_risk=functional_SNP_tobetested[functional_SNP_tobetested["P"] <= 5e-8].shape[0]
        not_gwas_risk=functional_SNP_tobetested[functional_SNP_tobetested["P"] > 5e-8].shape[0]
        is_not_gwas_risk_list=[is_gwas_risk, not_gwas_risk]
        is_not_gwas_risk_df=pd.DataFrame({"is_not_gwas_risk":is_not_gwas_risk_list})
    
        ###Load target roadmap states index
        if StatesIndex =="0":
            StatesIndexList = range(15)
        elif "_" in StatesIndex:
            StatesIndexList = [int(x) for x in StatesIndex.split("_")]
        else:
            StatesIndexList = [int(StatesIndex)]
            
        ###Calculate the percentage of SNPs locating in target roadmap states
        functional_SNP_tobetested_snps_ID=functional_SNP_tobetested["CHR_Position_A2_A1"]
        for i in StatesIndexList:
            cur_category = "E"+str(i)
            model_topSNP_in_cell_lines_functional_regions=[]
            model_topSNP_outof_cell_lines_functional_regions=[]
            percentage_model_topSNP_in_cell_lines_functional_regions=[]
            for cell_line in cell_lines:           
                SNP_in_func_regions,SNP_outof_func_regions = number_SNPs_in_out_of_func_regions(functional_SNP_tobetested_snps_ID,cell_line,cur_category)
                
                model_topSNP_in_cell_lines_functional_regions.append(len(SNP_in_func_regions))
                model_topSNP_outof_cell_lines_functional_regions.append(len(SNP_outof_func_regions))
                percentage_model_topSNP_in_cell_lines_functional_regions.append(len(SNP_in_func_regions)/(len(SNP_outof_func_regions)+len(SNP_in_func_regions)))
                
            model_enrichment_df = pd.DataFrame({"CellLines":cell_lines, "NumInFunc":model_topSNP_in_cell_lines_functional_regions, "NumOutofFunc":model_topSNP_outof_cell_lines_functional_regions,"PercentageInFunc":percentage_model_topSNP_in_cell_lines_functional_regions})
            final_df = pd.concat([model_enrichment_df,is_not_gwas_risk_df,maf_list_df])
            final_df.to_csv(wk+model_name+"_"+SnpNum+"_"+cur_category+"_percentage.csv",index=False)
        
if __name__ == "__main__":
    SnpNumsString=sys.argv[1]
    model=sys.argv[2]
    StatesIndex="7" #0 for 15 states. Otherwise, join by _, e.g. 6_7
    process(SnpNumsString, model, StatesIndex)
