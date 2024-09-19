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
    dis_df = pd.read_parquet("/work/long_lab/qli/Enformer_DTL/"+dis+"_dbSNPs_impute_summary_statistic_polyfun_MAF0.01.parquet")
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
    root_wk="/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg38_scores/"+batch+"/"
    if not os.path.exists("/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/"+batch+"/"):
        os.system("mkdir -p /work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/"+batch+"/")
    log=open("/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/log.txt","w")

    files=os.listdir(root_wk)
    for file_index in range(49):
        fw=open("/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/"+batch+"/"+str(file_index)+".200K.SNPs.scores.hg19.txt","w")
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
        DTL_df = pd.read_csv("/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/"+batch+"/"+str(file_index)+".200K.SNPs.scores.hg19.txt",sep=",",header=None)
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
        DTL_df_abs = hg19_DTL_scores_perchr.abs()
        DTL_df_abs.to_parquet("/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/"+batch+"/"+str(i)+".200K.SNPs.abs.scores.hg19.parquet", engine="pyarrow")
        

### Merge UKB, DTL, Downloaded ENF scores together        
def merge_UKB_DTL_ENF(dis,batch, scores_filenum, scores_m, logger):  
    for chr_ in range(1,23):
        SUMSTATS=pd.read_parquet("/work/long_lab/qli/Enformer_DTL/per_chr/"+dis+"_dbSNPs_impute_summary_statistic_polyfun_MAF0.01_chr"+str(chr_)+".parquet")
        SUMSTATS["CHR"] = SUMSTATS["CHR"].astype("str")
        SUMSTATS["BP"] = SUMSTATS["BP"].astype("str")
        SUMSTATS["A1"] = SUMSTATS["A1"].astype("str")
        SUMSTATS["A2"] = SUMSTATS["A2"].astype("str")
        SUMSTATS_set= set(SUMSTATS["CHR"].str.cat([SUMSTATS["BP"],SUMSTATS["A2"],SUMSTATS["A1"]],sep="_"))
        
        UKB_df=pd.read_parquet("/work/long_lab/qli/Enformer_TFs/polyfun/annotations/baselineLF2.2.UKB/baselineLF2.2.UKB."+str(chr_)+".annot.parquet")
        col_list = list(UKB_df.columns)
        col_list[3] = 'A2'
        col_list[4] = 'A1'
        UKB_df.columns = col_list

        UKB_df["CHR"] = UKB_df["CHR"].astype("str")
        UKB_df["BP"] = UKB_df["BP"].astype("str")
        UKB_df["A1"] = UKB_df["A1"].astype("str")
        UKB_df["A2"] = UKB_df["A2"].astype("str")

        UKB_df.index = UKB_df["CHR"].str.cat([UKB_df["BP"],UKB_df["A2"],UKB_df["A1"]],sep="_")
        UKB_df = UKB_df[~UKB_df.index.duplicated(keep='first')]
        UKB_df["INDEX"] = UKB_df.index
        UKB_df_SUMSTATS = UKB_df[UKB_df["INDEX"].isin(SUMSTATS_set)]
        del UKB_df_SUMSTATS["INDEX"]
        
        DTL_df_raw=pd.read_parquet("/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/"+dis+"_snps_hg19_scores/"+batch+"/"+str(chr_)+".200K.SNPs.scores.hg19.parquet")
        DTL_df=DTL_df_raw 
        DTL_df_abs = DTL_df.abs()
        DTL_df_abs["INDEX"] = DTL_df_abs.index
        DTL_df_abs_SUMSTATS = DTL_df_abs[DTL_df_abs["INDEX"].isin(SUMSTATS_set)]
        del DTL_df_abs_SUMSTATS["INDEX"]
        DTL_df_abs_SUMSTATS.to_parquet("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/"+str(chr_)+".DTL.NumberTracks.parquet")
        
        if(scores_m=="B"):
            DTL_columns_to_check = DTL_df_abs_SUMSTATS.columns[:]  # Select columns starting from some column
            for DTL_column in DTL_columns_to_check:
                DTL_column_mean = np.mean(DTL_df_abs_SUMSTATS[DTL_column])
                logger.info(str(chr_)+" "+DTL_column+ ",mean： "+str(DTL_column_mean))
                DTL_df_abs_SUMSTATS.loc[DTL_df_abs_SUMSTATS[DTL_column] >= DTL_column_mean, DTL_column] = 1
                DTL_df_abs_SUMSTATS.loc[DTL_df_abs_SUMSTATS[DTL_column] < DTL_column_mean, DTL_column] = 0
        
        ENF_df_raw = pd.read_parquet("/work/long_lab/qli/Enformer_TFs/HG19_scores/parquet/"+dis+"/"+str(chr_)+".parquet")    
        ENF_df=ENF_df_raw
        ENF_df_abs = ENF_df.abs()
        ENF_df_abs["INDEX"] = ENF_df_abs.index
        ENF_df_abs_SUMSTATS = ENF_df_abs[ENF_df_abs["INDEX"].isin(SUMSTATS_set)]
        del ENF_df_abs_SUMSTATS["INDEX"]
        ENF_df_abs_SUMSTATS.to_parquet("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/"+str(chr_)+".ENF.NumberTracks.parquet")
        
        if(scores_m=="B"):
            ENF_columns_to_check = list(ENF_df_abs_SUMSTATS.columns[:])  # Select columns starting from some column
            for ENF_column in ENF_columns_to_check:
                ENF_column_mean = np.mean(ENF_df_abs_SUMSTATS[ENF_column])
                logger.info(str(chr_)+" "+ENF_column+ ",mean： "+str(ENF_column_mean))
                ENF_df_abs_SUMSTATS.loc[ENF_df_abs_SUMSTATS[ENF_column] >= ENF_column_mean, ENF_column] = 1
                ENF_df_abs_SUMSTATS.loc[ENF_df_abs_SUMSTATS[ENF_column] < ENF_column_mean, ENF_column] = 0
        
        UKB_DTL= UKB_df_SUMSTATS.join(DTL_df_abs_SUMSTATS).fillna(0)
        UKB_DTL_ENF= UKB_DTL.join(ENF_df_abs_SUMSTATS).fillna(0)
        UKB_DTL_ENF['CHR'] = pd.to_numeric(UKB_DTL_ENF['CHR'], errors='coerce')
        UKB_DTL_ENF['BP'] = pd.to_numeric(UKB_DTL_ENF['BP'], errors='coerce')
        UKB_DTL_ENF_nodup = UKB_DTL_ENF.dropna(subset=['BP'])
        np.all(np.diff(UKB_DTL_ENF_nodup['BP'])>=0)
        UKB_DTL_ENF_nodup.iloc[:,5:]=UKB_DTL_ENF_nodup.iloc[:,5:].astype(int)
        UKB_DTL_ENF_nodup = UKB_DTL_ENF.drop_duplicates()
        UKB_DTL_ENF_nodup.to_parquet("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTLENFUKB/"+str(chr_)+".annot.parquet")
        
        #save l2 files
        with open("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTLENFUKB/"+str(chr_)+".l2.M","w") as f1:
            ss=str(int(UKB_DTL_ENF_nodup.iloc[:,5].sum()))
            for col in range(6,int(UKB_DTL_ENF_nodup.shape[1])):
                ss+="\t"+str(int(UKB_DTL_ENF_nodup.iloc[:,col].sum()))
            f1.write(ss+"\n") 

###Get single annotation resources
def get_single_annotation(dis,batch):
    if not os.path.exists("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/UKB/"):
        os.system("mkdir /work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/UKB/")
        
    if not os.path.exists("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTL/"):
        os.system("mkdir /work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTL/")
        
    if not os.path.exists("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/ENF/"):
        os.system("mkdir /work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/ENF/")
        
    if not os.path.exists("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/ld_annot/"):
        os.system("mkdir /work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/ld_annot/")

    for chr_ in range(1,23):
        UKB_DTL_ENF=pd.read_parquet("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTLENFUKB/"+str(chr_)+".annot.parquet")    
        UKB_DTL_ENF["BP"]=UKB_DTL_ENF["BP"].astype("int")
        UKB_DTL_ENF["CHR"]=UKB_DTL_ENF["CHR"].astype("int")
        DTL_cols=["CHR","SNP","BP","A2","A1"]
        DTL_cols_suff = [col for col in UKB_DTL_ENF.columns if 'DTL' in col]
        DTL_cols.extend(DTL_cols_suff)
        ENF_cols=["CHR","SNP","BP","A2","A1"]
        ENF_cols_suff = [col for col in UKB_DTL_ENF.columns if 'ENF' in col]
        ENF_cols.extend(ENF_cols_suff)
        UKB_cols=[col for col in UKB_DTL_ENF.columns if(('ENF' not in col) and ('DTL' not in col))]
        
        UKB_df=UKB_DTL_ENF[UKB_cols]
        ENF_df=UKB_DTL_ENF[ENF_cols]
        DTL_df=UKB_DTL_ENF[DTL_cols]
        
        UKB_df.to_parquet("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/UKB/"+str(chr_)+".annot.parquet")
        ENF_df.to_parquet("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/ENF/"+str(chr_)+".annot.parquet")
        DTL_df.to_parquet("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTL/"+str(chr_)+".annot.parquet")
        
        with open("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/UKB/"+str(chr_)+".l2.M","w") as f1:
            ss=str(int(UKB_df.iloc[:,5].sum()))
            for col in range(6,int(UKB_df.shape[1])):
                ss+="\t"+str(int(UKB_df.iloc[:,col].sum()))
            f1.write(ss+"\n") 
            
        with open("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/ENF/"+str(chr_)+".l2.M","w") as f1:
            ss=str(int(ENF_df.iloc[:,5].sum()))
            for col in range(6,int(ENF_df.shape[1])):
                ss+="\t"+str(int(ENF_df.iloc[:,col].sum()))
            f1.write(ss+"\n") 
            
        with open("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTL/"+str(chr_)+".l2.M","w") as f1:
            ss=str(int(DTL_df.iloc[:,5].sum()))
            for col in range(6,int(DTL_df.shape[1])):
                ss+="\t"+str(int(DTL_df.iloc[:,col].sum()))
            f1.write(ss+"\n") 

###Get double annotation resources
def get_double_annotations(dis,batch):
    if not os.path.exists("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/UKBENF/"):
        os.system("mkdir /work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/UKBENF/")
        
    if not os.path.exists("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/UKBDTL/"):
        os.system("mkdir /work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/UKBDTL/")
        
    if not os.path.exists("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTLENF/"):
        os.system("mkdir /work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTLENF/")
        
    if not os.path.exists("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTLENFUKB/"):
        os.system("mkdir /work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTLENFUKB/")
        
    if not os.path.exists("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/ld_annot/"):
        os.system("mkdir /work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/ld_annot/")

    for chr_ in range(1,23):
        UKB_DTL_ENF=pd.read_parquet("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTLENFUKB/"+str(chr_)+".annot.parquet")
        UKB_DTL_ENF["BP"]=UKB_DTL_ENF["BP"].astype("int")
        UKB_DTL_ENF["CHR"]=UKB_DTL_ENF["CHR"].astype("int")
        
        DTL_cols=["CHR","SNP","BP","A2","A1"]
        DTL_cols_suff = [col for col in UKB_DTL_ENF.columns if 'DTL' in col]
        DTL_cols.extend(DTL_cols_suff)
        ENF_cols=["CHR","SNP","BP","A2","A1"]
        ENF_cols_suff = [col for col in UKB_DTL_ENF.columns if 'ENF' in col]
        ENF_cols.extend(ENF_cols_suff)
        UKBENF_cols=[col for col in UKB_DTL_ENF.columns if(('ENF' not in col) and ('DTL' not in col))]
        UKBDTL_cols=[col for col in UKB_DTL_ENF.columns if(('ENF' not in col) and ('DTL' not in col))]
        
        UKBENF_cols.extend(ENF_cols_suff)
        UKBDTL_cols.extend(DTL_cols_suff)
        DTLENF_cols=["CHR","SNP","BP","A2","A1"]
        DTLENF_cols.extend(DTL_cols_suff)
        DTLENF_cols.extend(ENF_cols_suff)
        
        UKBENF_df=UKB_DTL_ENF[UKBENF_cols]
        UKBDTL_df=UKB_DTL_ENF[UKBDTL_cols]
        DTLENF_df=UKB_DTL_ENF[DTLENF_cols]
        
        UKBENF_df.to_parquet("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/UKBENF/"+str(chr_)+".annot.parquet")
        UKBDTL_df.to_parquet("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/UKBDTL/"+str(chr_)+".annot.parquet")
        DTLENF_df.to_parquet("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTLENF/"+str(chr_)+".annot.parquet")
        
        with open("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/UKBENF/"+str(chr_)+".l2.M","w") as f1:
            ss=str(int(UKBENF_df.iloc[:,5].sum()))
            for col in range(6,int(UKBENF_df.shape[1])):
                ss+="\t"+str(int(UKBENF_df.iloc[:,col].sum()))
            f1.write(ss+"\n") 
            
        with open("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/UKBDTL/"+str(chr_)+".l2.M","w") as f1:
            ss=str(int(UKBDTL_df.iloc[:,5].sum()))
            for col in range(6,int(UKBDTL_df.shape[1])):
                ss+="\t"+str(int(UKBDTL_df.iloc[:,col].sum()))
            f1.write(ss+"\n") 
            
        with open("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/DTLENF/"+str(chr_)+".l2.M","w") as f1:
            ss=str(int(DTLENF_df.iloc[:,5].sum()))
            for col in range(6,int(DTLENF_df.shape[1])):
                ss+="\t"+str(int(DTLENF_df.iloc[:,col].sum()))
            f1.write(ss+"\n") 

def write_cmd(dis, batch):
    for chr_ in range(1,23):
        cmd_file = open("/work/long_lab/qli/Enformer_DTL/HG19_UKB_DTL_Mto1_ENF_Mto1_ABS_scores_10M/"+dis+"/"+batch+"/ld_annot/ld_annot_"+str(chr_)+".cmd","w")
        cmd_file.write('#!/bin/bash\n')
        cmd_file.write('#SBATCH --job-name=ldscore_'+str(chr_)+'\n') 
        cmd_file.write('#SBATCH --error='+str(chr_)+'.error\n')
        cmd_file.write('#SBATCH --out='+str(chr_)+'.out\n')
        cmd_file.write('#SBATCH --mem=48G\n')
        cmd_file.write('#SBATCH --nodes=1\n')
        cmd_file.write('#SBATCH --ntasks=1\n')
        cmd_file.write('#SBATCH --cpus-per-task=1\n')
        cmd_file.write('#SBATCH --time=7-0:0:0\n')
        cmd_file.write('#SBATCH --partition=cpu2019,cpu2021,cpu2022,cpu2023,theia,mtst\n')
        cmd_file.write('python /work/long_lab/qli/Enformer_TFs/polyfun/compute_ldscores_from_ld.py --annot ../UKB/'+str(chr_)+'.annot.parquet --ld-dir /work/long_lab/qli/Enformer_TFs/polyfun/LD_cache_hg19/ --ukb  --out ../UKB/'+str(chr_)+'.l2.ldscore.parquet \n')
        cmd_file.write('python /work/long_lab/qli/Enformer_TFs/polyfun/compute_ldscores_from_ld.py --annot ../DTL/'+str(chr_)+'.annot.parquet --ld-dir /work/long_lab/qli/Enformer_TFs/polyfun/LD_cache_hg19/ --ukb --out ../DTL/'+str(chr_)+'.l2.ldscore.parquet \n')
        cmd_file.write('python /work/long_lab/qli/Enformer_TFs/polyfun/compute_ldscores_from_ld.py --annot ../ENF/'+str(chr_)+'.annot.parquet --ld-dir /work/long_lab/qli/Enformer_TFs/polyfun/LD_cache_hg19/ --ukb --out ../ENF/'+str(chr_)+'.l2.ldscore.parquet \n')
        cmd_file.write('python /work/long_lab/qli/Enformer_TFs/polyfun/compute_ldscores_from_ld.py --annot ../UKBENF/'+str(chr_)+'.annot.parquet --ld-dir /work/long_lab/qli/Enformer_TFs/polyfun/LD_cache_hg19/ --ukb  --out ../UKBENF/'+str(chr_)+'.l2.ldscore.parquet \n')
        cmd_file.write('python /work/long_lab/qli/Enformer_TFs/polyfun/compute_ldscores_from_ld.py --annot ../UKBDTL/'+str(chr_)+'.annot.parquet --ld-dir /work/long_lab/qli/Enformer_TFs/polyfun/LD_cache_hg19/ --ukb --out ../UKBDTL/'+str(chr_)+'.l2.ldscore.parquet \n')
        cmd_file.write('python /work/long_lab/qli/Enformer_TFs/polyfun/compute_ldscores_from_ld.py --annot ../DTLENF/'+str(chr_)+'.annot.parquet --ld-dir /work/long_lab/qli/Enformer_TFs/polyfun/LD_cache_hg19/ --ukb --out ../DTLENF/'+str(chr_)+'.l2.ldscore.parquet \n')
        cmd_file.write('python /work/long_lab/qli/Enformer_TFs/polyfun/compute_ldscores_from_ld.py --annot ../DTLENFUKB/'+str(chr_)+'.annot.parquet --ld-dir /work/long_lab/qli/Enformer_TFs/polyfun/LD_cache_hg19/ --ukb --out ../DTLENFUKB/'+str(chr_)+'.l2.ldscore.parquet \n')
        cmd_file.close()


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
    scores_txtToparquet(dis,batch,scores_filenum, logger)
    merge_UKB_DTL_ENF(dis,batch,scores_filenum,scores_m, logger)
    print("Finish Merge three sets of scores")
    get_single_annotation(dis,batch)
    print("Finish one set of scores")
    get_double_annotations(dis,batch)
    print("Finish two sets of scores")
    write_cmd(dis, batch)
    logger.info('Program finished')
    
    
if __name__ == '__main__':
    main()
