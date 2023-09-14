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
    
def number_SNPs_in_out_of_func_regions(target_SNPs_list,prefix,category):
    flank=0
    SNP_in_outof_func_regions=[]
    
    chr_pos_start_lists_sorted,chr_pos_start_end_dict = chr_pos_strat_end_lists(prefix,category)
    for snp in target_SNPs_list:
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
                SNP_in_outof_func_regions.append(1)  #In functional region
                print(snp,fs,fe,bs,be)
            else:
                SNP_in_outof_func_regions.append(0)

    return SNP_in_outof_func_regions

def process(model_name, StatesIndex):
    wk="/work/long_lab/qli/Breat_TFs_SNPs/Enhancer_Promoters_Annotations/roadmap/percentage_res/"
    cell_lines=["E027","E028","E029","E055","E066","E119","E121","E124","E125","E126"]
   
    lo=LiftOver('hg19','hg38')  #'hg19', 'hg38'
    SNP_tobetested = pd.read_csv("/work/long_lab/qli/Enformer_DTL/Manuscript_Figures_Tables/"+model_name+"_breast.1_22.cs.1e-6.manual2.completeccvs.csv")
    hg38_pos_list=[]
    
    for row_index in range(SNP_tobetested.shape[0]):
        chr_=SNP_tobetested.iloc[row_index,0]
        bp_hg19=SNP_tobetested.iloc[row_index,2]
        new = lo.convert_coordinate("chr"+str(chr_), int(bp_hg19))
        if new is not None and (len(new)>0):
            hg38_pos = str(new[0][1])
        else:
            hg38_pos = np.nan
        hg38_pos_list.append(hg38_pos)
    
    SNP_tobetested["BP_hg38"] = hg38_pos_list
    SNP_tobetested_dropna = SNP_tobetested.dropna()
    print(SNP_tobetested_dropna.shape)
    
    SNP_tobetested_dropna["BP_hg38"]=SNP_tobetested_dropna["BP_hg38"].astype("str")
    SNP_tobetested_dropna["CHR"]=SNP_tobetested_dropna["CHR"].astype("str")
    SNP_tobetested_dropna["CHR_Position_A2_A1"]=SNP_tobetested_dropna[["CHR","BP_hg38","A2","A1"]].agg('_'.join, axis=1)
    print(SNP_tobetested_dropna.shape)
   
   
    ###Load target roadmap states index
    if StatesIndex =="0":
        StatesIndexList = range(15)
    elif "_" in StatesIndex:
        StatesIndexList = [int(x) for x in StatesIndex.split("_")]
    else:
        StatesIndexList = [int(StatesIndex)]
        
    ###Calculate the percentage of SNPs locating in target roadmap states
    for i in StatesIndexList:
        cur_category = "E"+str(i)
      
        cur_category_cell_lines_in_out_df = SNP_tobetested_dropna[["CHR_Position_A2_A1","P","PIP","CREDIBLE_SET"]]
        
        model_topSNP_in_cell_lines_functional_regions=[]
        model_topSNP_outof_cell_lines_functional_regions=[]
        percentage_model_topSNP_in_cell_lines_functional_regions=[]
        for cell_line in cell_lines:           
            SNP_in_outof_func_regions_list = number_SNPs_in_out_of_func_regions(list(SNP_tobetested_dropna["CHR_Position_A2_A1"]),cell_line,cur_category)
            print(SNP_in_outof_func_regions_list)
            SNP_in_outof_func_regions_df = pd.DataFrame({cell_line:SNP_in_outof_func_regions_list})
            cur_category_cell_lines_in_out_df = pd.concat([cur_category_cell_lines_in_out_df, SNP_in_outof_func_regions_df],axis=1)
            
        cur_category_cell_lines_in_out_df.to_csv(wk+model_name+"_crediblesets_"+cur_category+"_percentage.completeccvs.csv",index=False)
            
        
if __name__ == "__main__":
    model=sys.argv[1]
    StatesIndex="7" #0 for 15 states. Otherwise, join by _, e.g. 6_7
    process(model, StatesIndex)
