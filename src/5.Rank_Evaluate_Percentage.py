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
import logging

logging.basicConfig(filename='process.log', level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
    
    
def chr_pos_strat_end_lists(dis, CellLine,ChromState):
    chr_pos_start_lists=[]
    chr_pos_start_lists_sorted=[]
    chr_pos_start_end_dict={}
    for i in range(22):
        chr_pos_start_lists.append(list())
    if dis=="breast":
        f=open("/work/long_lab/qli/postdoc/Cancer_GWAS_SS/Enhancer_Promoters_Annotations/roadmap_enhancer_bed/AllDenseBedHg19/"+CellLine+"_15_coreMarks_dense.bed","r")
    elif dis=="prostate":
        f=open("/work/long_lab/qli/postdoc/Cancer_GWAS_SS/Enhancer_Promoters_Annotations/PMID34513844_15chromatin_states/"+CellLine+"_15states_joint_model_dense.bed","r")
    
    header=f.readline()
    line=f.readline().strip()
    while line:
        if (not line.startswith("chrX")) and (not line.startswith("chrY")):
            line_arr = line.split("\t")
            if line_arr[3].startswith(str(ChromState)+"_"): 
                if re.search(r'\d{1,2}',line_arr[0]) is not None:
                    chr_num=int(re.search(r'\d{1,2}',line_arr[0]).group())
                    chr_pos_start_lists[chr_num-1].append(int(line_arr[1]))  #[[start1, start2, ..., startN],[]]
                    key=str(chr_num)+"_"+line_arr[1]
                    chr_pos_start_end_dict[key]=key+"_"+line_arr[2] #dic{key=chr_start, value=chr_start_end}
        else:
            break
        line=f.readline().strip()
    f.close()
    state_location_count=0
    for i in range(22):
        chr_pos_start_lists_sorted.append(sorted(chr_pos_start_lists[i]))
        state_location_count+=len(chr_pos_start_lists[i])
    print("Finished collecting locations for "+str(ChromState)+". Total state locations "+str(state_location_count))
    return chr_pos_start_lists_sorted,chr_pos_start_end_dict
    
    
def SNPs_percent(K,N_K):
    return(len(K), len(K)+len(N_K), len(K)/len(K+N_K))
    
def number_SNPs_in_out_of_func_regions(dis, target_SNPs_set,CellLine,ChromState):
    records_df=pd.DataFrame(columns=['CellLine', 'snp', 'fs', 'fe', 'bs', 'be'])
    flank=0
    SNP_in_func_regions=[]
    SNP_outof_func_regions=[]
    chr_pos_start_lists_sorted,chr_pos_start_end_dict = chr_pos_strat_end_lists(dis, CellLine, ChromState) #prefix=CellLineName; category=chromatin_status
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
                #print(snp,fs,fe,bs,be)
                new_row = pd.DataFrame({'CellLine': [CellLine], 'snp': [snp], 'fs': [fs], 'fe': [fe], 'bs': [bs], 'be': [be]})
                records_df = pd.concat([records_df, new_row], ignore_index=True)
            else:
                SNP_outof_func_regions.append(snp)  
    return SNP_in_func_regions,SNP_outof_func_regions,records_df


def process(dis, batch, SnpNumsString , model, StatesIndex, log):
    wk="/work/long_lab/qli/postdoc/Cancer_GWAS_SS/Enhancer_Promoters_Annotations/roadmap/percentage_res/"
    model_prefix=dis+"_"+model+"_"+batch
    if dis=="breast":
        cell_lines=["E027","E028","E029","E055","E066","E119","E121","E124","E125","E126"]
    elif dis=="prostate":
        cell_lines=["prostate"]
    
    ###Load functional SNPs
    model_output_df_raw=pd.DataFrame()
    for chr_ in range(1,23):
        perchr=pd.read_parquet("/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/"+batch+"/"+str(chr_)+".200K.SNPs.abs.scores.hg19.parquet")
        model_output_df_raw=pd.concat([model_output_df_raw, perchr])
    model_output_df_raw_nodup = model_output_df_raw.drop_duplicates()
    model_output_df=model_output_df_raw_nodup.abs()
    log.info('Read functional %d SNPs.', model_output_df.shape[0]) 
    ###Load GWAS summary statistic file
    gwas_sumstats = pd.read_csv("/work/long_lab/qli/postdoc/Cancer_GWAS_SS/"+dis+"_dbSNPs_impute_summary_statistic_polyfun.txt",sep="\t")
    gwas_sumstats['CHR']=gwas_sumstats['CHR'].astype("str")
    gwas_sumstats['POSITION']=gwas_sumstats['POSITION'].astype('str')
    gwas_sumstats['ID']=gwas_sumstats[['CHR', 'POSITION','A2','A1']].agg('_'.join, axis=1)
    gwas_sumstats.index=gwas_sumstats['ID']
    gwas_sumstats_risk = gwas_sumstats[gwas_sumstats["PVALUE"]<5e-8]
    gwas_sumstats_risk_set = set(gwas_sumstats_risk['ID'])
    log.info('Read GWAS SS %d SNPs.', gwas_sumstats.shape[0])
    
    ###Identify top SnpNum SNPs
    SnpNumsList=[]
    if "," in SnpNumsString:
        SnpNumsList.extend(SnpNumsString.split(","))
    else:
        SnpNumsList.append(SnpNumsString)
    print(SnpNumsList)
    
    model_output_df_mean = pd.DataFrame(model_output_df.iloc[:, :].mean(axis=1))
    model_output_df_mean.columns=["MeanTFs"] #
    model_output_df_mean["AbsMeanTFs"] = model_output_df_mean["MeanTFs"].abs()
    model_output_df_mean_sorted = model_output_df_mean.sort_values(by="AbsMeanTFs",ascending=False)
    
    for SnpNum in SnpNumsList:
        model_output_df_mean_sorted_topSnpNum = model_output_df_mean_sorted.iloc[:int(SnpNum),:]
        ###Number of MAF SNPs
        model_output_df_mean_sorted_topSnpNum_MAF=pd.merge(model_output_df_mean_sorted_topSnpNum, gwas_sumstats["A1FREQ"], left_index=True, right_index=True, how='inner')
        model_output_df_mean_sorted_topSnpNum_MAF["A1FREQ"]=model_output_df_mean_sorted_topSnpNum_MAF["A1FREQ"].astype("float")
        model_output_df_mean_sorted_topSnpNum_MAF["MAF"]=list(np.where(model_output_df_mean_sorted_topSnpNum_MAF["A1FREQ"]>0.5, 1-model_output_df_mean_sorted_topSnpNum_MAF["A1FREQ"], model_output_df_mean_sorted_topSnpNum_MAF["A1FREQ"]))
        model_output_df_mean_sorted_topSnpNum_MAF1 = model_output_df_mean_sorted_topSnpNum_MAF[model_output_df_mean_sorted_topSnpNum_MAF["MAF"]<0.01].shape[0]
        model_output_df_mean_sorted_topSnpNum_MAF2 = model_output_df_mean_sorted_topSnpNum_MAF[(model_output_df_mean_sorted_topSnpNum_MAF["MAF"]>=0.01) & (model_output_df_mean_sorted_topSnpNum_MAF["MAF"]<0.05)].shape[0]
        model_output_df_mean_sorted_topSnpNum_MAF3 = model_output_df_mean_sorted_topSnpNum_MAF[(model_output_df_mean_sorted_topSnpNum_MAF["MAF"]>=0.05) & (model_output_df_mean_sorted_topSnpNum_MAF["MAF"]<0.1)].shape[0]
        model_output_df_mean_sorted_topSnpNum_MAF4 = model_output_df_mean_sorted_topSnpNum_MAF[(model_output_df_mean_sorted_topSnpNum_MAF["MAF"]>=0.1) & (model_output_df_mean_sorted_topSnpNum_MAF["MAF"]<0.2)].shape[0]
        model_output_df_mean_sorted_topSnpNum_MAF5 = model_output_df_mean_sorted_topSnpNum_MAF[model_output_df_mean_sorted_topSnpNum_MAF["MAF"]>=0.2].shape[0]
        maf_list=[model_output_df_mean_sorted_topSnpNum_MAF1,model_output_df_mean_sorted_topSnpNum_MAF2,model_output_df_mean_sorted_topSnpNum_MAF3,model_output_df_mean_sorted_topSnpNum_MAF4,model_output_df_mean_sorted_topSnpNum_MAF5]
        maf_list_df=pd.DataFrame({"MAF":maf_list})
        
        ###Number of GWAS risk SNPs
        model_output_df_mean_sorted_topSnpNum_snps_ID = list(model_output_df_mean_sorted_topSnpNum.index)
        is_gwas_risk=len(set(model_output_df_mean_sorted_topSnpNum_snps_ID).intersection(gwas_sumstats_risk_set))
        not_gwas_risk=len(set(model_output_df_mean_sorted_topSnpNum_snps_ID).difference(gwas_sumstats_risk_set))
        is_not_gwas_risk_list=[is_gwas_risk, not_gwas_risk]
        is_not_gwas_risk_df=pd.DataFrame({"is_not_gwas_risk":is_not_gwas_risk_list})
    
        ###Load target roadmap states index
        '''
        # promoter_ID_list=["E1","E2","E10","E11"] 1_2_10_11
        # promoter_enhancer_ID_list=["E1","E2","E6","E7","E10","E11","E12"]
        # enhancer_ID_list=["E6","E7","E11","E12"] 6_7_11_12
        '''
        if StatesIndex =="0":
            StatesIndexList = range(15)
        elif "_" in StatesIndex:
            StatesIndexList = [int(x) for x in StatesIndex.split("_")]
        else:
            StatesIndexList = [int(StatesIndex)]
            
        ###Calculate the percentage of SNPs locating in target roadmap states / GWAS risk SNPs
        for i in StatesIndexList:
            ChromState = str(i)
            model_topSNP_in_cell_lines_functional_regions=[]
            model_topSNP_outof_cell_lines_functional_regions=[]
            percentage_model_topSNP_in_cell_lines_functional_regions=[]
            records_df_AllCellLine=pd.DataFrame(columns=['CellLine', 'snp', 'fs', 'fe', 'bs', 'be'])
            for CellLine in cell_lines:           
                SNP_in_func_regions,SNP_outof_func_regions,records_df_perCellLine = number_SNPs_in_out_of_func_regions(dis, model_output_df_mean_sorted_topSnpNum_snps_ID,CellLine,ChromState)
                records_df_AllCellLine=pd.concat([records_df_AllCellLine,records_df_perCellLine])
                model_topSNP_in_cell_lines_functional_regions.append(len(SNP_in_func_regions))
                model_topSNP_outof_cell_lines_functional_regions.append(len(SNP_outof_func_regions))
                percentage_model_topSNP_in_cell_lines_functional_regions.append(len(SNP_in_func_regions)/(len(SNP_outof_func_regions)+len(SNP_in_func_regions)))
            model_enrichment_df = pd.DataFrame({"CellLines":cell_lines, "NumInFunc":model_topSNP_in_cell_lines_functional_regions, "NumOutofFunc":model_topSNP_outof_cell_lines_functional_regions,"PercentageInFunc":percentage_model_topSNP_in_cell_lines_functional_regions})
            final_df = pd.concat([model_enrichment_df,is_not_gwas_risk_df,maf_list_df])
            final_df.to_csv(wk+model_prefix+"_"+str(SnpNum)+"_ChromState_"+str(ChromState)+"_percentage.csv",index=False)
            records_df_AllCellLine.to_csv(wk+model_prefix+"_"+str(SnpNum)+"_ChromState_"+str(ChromState)+"_records.csv",index=False)
            log.info('Finished %s for SNP sets %s.', ChromState, str(SnpNum))
            

if __name__ == "__main__":
    SnpNumsString=sys.argv[1]
    model=sys.argv[2] #DTL ENF
    dis=sys.argv[3]
    batch=sys.argv[4]
    StatesIndex="6_7" #0 for 15 states. Otherwise, join by _, e.g. 6_7
    log = logging.getLogger('my_logger')
    process(dis, batch, SnpNumsString ,model, StatesIndex,log)

