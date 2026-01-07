import sys
import os
import pandas as pd
import numpy as np
import json
import pdb
import re
from pyliftover import LiftOver
lo=LiftOver('hg38','hg19')  #'hg19', 'hg38'


'''
1. Select topX SNPs (topX+100, to avoid unsuccessful convert from hg38 to hg19)
2. Conver to hg38 to hg19 ()
3. Calculate the % of SNPs locating in enhancer
'''

def hg38tohg19(df):
	hg19_pos_list=[]
	for row_index in range(df.shape[0]):
		chr_num = df.iloc[row_index, 0]
		pos_hg38 = df.iloc[row_index, 1]
		new = lo.convert_coordinate("chr"+chr_num, int(pos_hg38))
		pos_hg19="0" #default as 0
		if((new is not None) and (len(new)>0)):
			pos_hg19=str(new[0][1])
		hg19_pos_list.append(pos_hg19)
	if(len(hg19_pos_list)==df.shape[0]):
		print(f'Length match after converting from hg38 to hg19')
		return hg19_pos_list
	else:
		print(f'Length mismatch. Please check code and input data')
		exit()

def topX_SNPs(folder_path, scores_file_num, topX_list):
    hg19_DTL_list = []
    for file_index in range(int(scores_file_num)):
        file_path = os.path.join(folder_path, f"{file_index}.200K.SNPs.scores.txt")
        # Read file
        DTL_df = pd.read_csv(file_path, sep=",", header=0)
        # Remove invalid SNPs
        DTL_df = DTL_df[~DTL_df["SNP_ID"].str.contains("nan", na=False)]
        # Split SNP_ID
        DTL0_df = DTL_df["SNP_ID"].str.split("_", expand=True)
        DTL0_df.columns = ["CHR", "BP_hg38", "A2", "A1"]
        # Liftover hg38 → hg19
        DTL0_df["BP_hg19"] = hg38tohg19(DTL0_df[["CHR", "BP_hg38"]])
        # Build hg19 SNP ID
        hg19_index = DTL0_df["CHR"].str.cat([DTL0_df["BP_hg19"].astype(str),DTL0_df["A2"],DTL0_df["A1"],],sep="_")
        # Prepare score dataframe
        DTL_df["hg19_index"]=hg19_index
        DTL_df=DTL_df[~DTL_df["hg19_index"].str.contains("_0_")] # remove SNP, which hg19 position is 0
        DTL_df.index = DTL_df["hg19_index"]
        DTL_df_ready = DTL_df.drop(columns=["SNP_ID", "hg19_index"])
        # Remove duplicated SNPs
        DTL_df_ready = DTL_df_ready[~DTL_df_ready.index.duplicated(keep="first")]
        # Compute absolute mean
        DTL_df_ready["Abs_mean"] = (DTL_df_ready.abs().mean(axis=1))
        hg19_DTL_list.append(DTL_df_ready)
        print(f"Finished reading score file {file_index}")
    # Combine all files
    hg19_DTL_scores = pd.concat(hg19_DTL_list)
    # Sort by Abs_mean
    hg19_DTL_scores_sorted = (hg19_DTL_scores.sort_values("Abs_mean", ascending=False).reset_index().rename(columns={"index": "SNP_ID_hg19"}))
    print(f'Finish sorting by abs mean')
	# Return topX SNPs as dict
    return {int(topX): hg19_DTL_scores_sorted.iloc[:int(topX)] for topX in topX_list}

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
    
    
def chr_pos_strat_end_lists(CellLine,ChromState):
    chr_pos_start_lists=[]
    chr_pos_start_lists_sorted=[]
    chr_pos_start_end_dict={}
    for i in range(22):
        chr_pos_start_lists.append(list())
    f=open("/work/long_lab/qli/postdoc/Cancer_GWAS_SS/Enhancer_Promoters_Annotations/roadmap_enhancer_bed/AllDenseBedHg19/"+CellLine+"_15_coreMarks_dense.bed","r")
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
    
def number_SNPs_in_out_of_func_regions(target_SNPs_set,CellLine,ChromState):
    records_df=pd.DataFrame(columns=['CellLine', 'snp', 'fs', 'fe', 'bs', 'be'])
    flank=0
    SNP_in_func_regions=[]
    SNP_outof_func_regions=[]
    chr_pos_start_lists_sorted,chr_pos_start_end_dict = chr_pos_strat_end_lists(CellLine, ChromState) #prefix=CellLineName; category=chromatin_status
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
            if(front_start_end is not None) and (behind_start_end is not None):
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

 
def main():
    '''
    # promoter_ID_list=["E1","E2","E10","E11"] 1_2_10_11
    # promoter_enhancer_ID_list=["E1","E2","E6","E7","E10","E11","E12"]
    # enhancer_ID_list=["E6","E7","E11","E12"] 6_7_11_12
    '''
    
	folder_path="/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/1_revision_DTL/CNN_breast_hg38_scores/"
	scores_file_num=51 # [0...51)
	topX_list=[1e6, 1.5e6, 2e6]
	topX_dfs = topX_SNPs(folder_path, scores_file_num, topX_list)
    
    StatesIndex="6_7"
    cell_lines=["prostate","E027","E028","E029","E055","E066","E119","E121","E124","E125","E126"]
    output_path="/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/1_revision_DTL/CNN_breast_hg19_E6_E7_percent/"
    model_prefix="CNN"
    dis="breast"
    
    if StatesIndex =="0":
        StatesIndexList = range(15)
    elif "_" in StatesIndex:
        StatesIndexList = [int(x) for x in StatesIndex.split("_")]
    else:
        StatesIndexList = [int(StatesIndex)]
        
    ###Calculate the percentage of SNPs locating in target roadmap states / GWAS risk SNPs
    for topX_key in topX_dfs.keys():
        topX_df=topX_dfs.get(topX_key)
        topX_SNPs_ID_hg19=list(topX_df["hg19_index"])
        for i in StatesIndexList:
            ChromState = str(i)
            model_topSNP_in_cell_lines_functional_regions=[]
            model_topSNP_outof_cell_lines_functional_regions=[]
            percentage_model_topSNP_in_cell_lines_functional_regions=[]
            records_df_AllCellLine=pd.DataFrame(columns=['CellLine', 'snp', 'fs', 'fe', 'bs', 'be'])
            for CellLine in cell_lines:           
                SNP_in_func_regions,SNP_outof_func_regions,records_df_perCellLine = number_SNPs_in_out_of_func_regions(topX_SNPs_ID_hg19, CellLine, ChromState)
                records_df_AllCellLine=pd.concat([records_df_AllCellLine,records_df_perCellLine])
                model_topSNP_in_cell_lines_functional_regions.append(len(SNP_in_func_regions))
                model_topSNP_outof_cell_lines_functional_regions.append(len(SNP_outof_func_regions))
                percentage_model_topSNP_in_cell_lines_functional_regions.append(len(SNP_in_func_regions)/(len(SNP_outof_func_regions)+len(SNP_in_func_regions)))
            model_enrichment_df = pd.DataFrame({"CellLines":cell_lines, "NumInFunc":model_topSNP_in_cell_lines_functional_regions, "NumOutofFunc":model_topSNP_outof_cell_lines_functional_regions,"PercentageInFunc":percentage_model_topSNP_in_cell_lines_functional_regions})
            model_enrichment_df.to_csv(output_path+model_prefix+"_"+str(topX_key)+"_ChromState_"+str(ChromState)+"_percentage.csv",index=False)
            records_df_AllCellLine.to_csv(output_path+model_prefix+"_"+str(topX_key)+"_ChromState_"+str(ChromState)+"_records.csv",index=False)
        
if __name__ == '__main__':
    main()
