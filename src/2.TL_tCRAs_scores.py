#!/usr/bin/env python
# coding: utf-8

'''
Rewrite on Nov 8th 2022. 
-Usage:
Calculate scores from models after transfer learning

-Changlog:
Reformat file to contain only diff scores calculation
'''

# In[1]:
import sonnet as snt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from datetime import datetime
import glob
import json
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
from utils import *
import functools
import pyfaidx
import sys
#import tfenformer_breast
import tfenformer_prostate
import time
import configparser

config=configparser.ConfigParser()
config.read("2.enformer-usage_SNPs_q_CC_DTL.ini")

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]='false' # Easier debugging of OOM

# Set the GPU index you want to use
gpu_index = 0
# Get the list of available devices
devices = tf.config.list_physical_devices('GPU')
# Set the desired GPU as visible
tf.config.experimental.set_visible_devices(devices[gpu_index], 'GPU')

assert snt.__version__.startswith('2.0')
print(tf.__version__)

###Set up disease and fasta
dis=config.get("Section1",'dis')
num_point_tobe_checked=config.get("Section1",'num_point_tobe_checked')
checkpoint_path = './ModelTraining_OldDGX/human_tfrecords_'+dis+'/checkpoint/checkpoint'+num_point_tobe_checked      
SNPs_per_file="200K"
dis_dict={"breast":275,"lung":136,"prostate":357}

fasta_file = './ModelTraining_OldDGX/hg38.ml.fa'
fasta_extractor = FastaStringExtractor(fasta_file)
SEQUENCE_LENGTH=196608

###Read SNPs that will be used to calcualte scores
chrom_item = sys.argv[1] #"1"
print("Index in 200K: "+chrom_item)
f=open("./ModelTraining_OldDGX/"+dis+"_snps_hg38/"+chrom_item+"."+SNPs_per_file+".txt","r")
SNPs_list=[]
line=f.readline()
while line: 
    SNPs_list.append(line.strip())
    line=f.readline()
f.close()

if not os.path.exists("./ModelTraining_OldDGX/"+dis+"_snps_hg38_scores/"+num_point_tobe_checked+"/"):
    os.mkdir("./ModelTraining_OldDGX/"+dis+"_snps_hg38_scores/"+num_point_tobe_checked+"/")
fw=open("./ModelTraining_OldDGX/"+dis+"_snps_hg38_scores/"+num_point_tobe_checked+"/"+chrom_item+"."+SNPs_per_file+".SNPs.scores.txt","w")
col_index_for_head = [str(x) for x in range(1,dis_dict.get(dis))]
fw.write("SNP_ID,"+",".join(col_index_for_head)+"\n") ###Write out the header for scores files

###Prepare model
if dis=="breast":
  model = tfenformer_breast.Enformer(channels=1536,num_heads=8,num_transformer_layers=11,pooling_type='attention')
elif dis=="prostate":
  model = tfenformer_prostate.Enformer(channels=1536,num_heads=8,num_transformer_layers=11,pooling_type='attention')                 
                          
checkpoint = tf.train.Checkpoint(module=model)                    
latest = tf.train.latest_checkpoint(checkpoint_path)
print(latest)
status = checkpoint.restore(latest)


count=0
for SNP in SNPs_list:
    start_time=time.time()
    chr_,pos,ref,alt=SNP.split("_")
    
    if int(pos) <= 200000: ##Cannot predict SNPs accurately if its position is too front as FASTA file starts from around 100K position. 
        continue
        
    variant = kipoiseq.Variant('chr'+chr_, int(pos), ref, alt)  # @param
    interval = kipoiseq.Interval(variant.chrom, variant.start, variant.start).resize(SEQUENCE_LENGTH)
    seq_extractor = kipoiseq.extractors.VariantSeqExtractor(reference_sequence=fasta_extractor)
    center = interval.center() - interval.start

    try: 
        reference = seq_extractor.extract(interval, [], anchor=center)
        alternate = seq_extractor.extract(interval, [variant], anchor=center)

        ## Make predictions for the refernece and alternate allele
        # reference_prediction = model.predict_on_batch(one_hot_encode(reference)[np.newaxis])['human'][0]
        # alternate_prediction = model.predict_on_batch(one_hot_encode(alternate)[np.newaxis])['human'][0]
        reference_prediction = model(one_hot_encode(reference)[np.newaxis], is_training=False)['human'][0] #896, target dim
        alternate_prediction = model(one_hot_encode(alternate)[np.newaxis], is_training=False)['human'][0] #896, target dim
    except Exception as e:
        print("Error:", e)
        continue
    
    variant_pred_pos=reference_prediction.shape[0]//2-1 #447
    
    diff_pred_score=np.matrix(np.array(alternate_prediction[variant_pred_pos:(variant_pred_pos+2),:])-np.array(reference_prediction[variant_pred_pos:(variant_pred_pos+2),:]))
    diff_pred_score_mean=np.mean(diff_pred_score, axis=0).tolist()
    end_time=time.time()
    time_cost=time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))
    diff_pred_score_str = [f"{x:.6f}" for x in diff_pred_score_mean[0]]
    fw.write(SNP+","+",".join(diff_pred_score_str)+"\n")
    count+=1
    if not count % 200: 
        fw.flush()
fw.close()
