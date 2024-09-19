#!/usr/bin/env python
# coding: utf-8

'''
Initilize 20230630

Usage: Calculate person R for each track across all position in test sets
'''

# In[1]:
import sonnet as snt
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from datetime import datetime
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]='false'
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
import tfenformer_breast
import enformer


assert snt.__version__.startswith('2.0')
print(tf.__version__)


###Hyperparameters Definition
hyper_paras=Hyper_paras()
hyper_paras.evaluate_steps = 1937  #1,937, should test all samples in test datasets
tissue="breast"


#######For DTL
###Load test datasets
#human_test_dataset = get_dataset('human', 'test', tissue).batch(hyper_paras.batch_size).repeat(hyper_paras.epoch).prefetch(hyper_paras.batch_size)

# ###Prepare DTL model
# model = tfenformer_breast.Enformer(channels=1536,
                          # num_heads=8,
                          # num_transformer_layers=11,
                          # pooling_type='attention')
                          
# checkpoint = tf.train.Checkpoint(module=model)      
# checkpoint_path = '/work/long_lab/qli/Enformer_DTL/FromDGX/DTL_202211/'+tissue+'/checkpoint'                    
# latest = tf.train.latest_checkpoint(checkpoint_path)
# print(latest)
# status = checkpoint.restore(latest)


# ###Evaluate final model on test datasets        
# @tf.function
# def predict(x):
    # return model(x, is_training=False)['human']

# for i, batch in tqdm(enumerate(human_test_dataset)):
    # if i < hyper_paras.evaluate_steps: 
        # print("Saving tensor values for "+str(i+1))
        # np.save(str(i+1)+".true.npy", batch['target'].numpy())
        # np.save(str(i+1)+".predict.npy", predict(batch['sequence']).numpy())
        # print("Finish saving tensor values for "+str(i+1))

#######For ENF
###Load test datasets
human_test_dataset = get_dataset('humanENF', 'test', tissue).batch(hyper_paras.batch_size).repeat(hyper_paras.epoch).prefetch(hyper_paras.batch_size)

###Prepare ENF model
model = enformer.Enformer(channels=1536,
                          num_heads=8,
                          num_transformer_layers=11,
                          pooling_type='attention')
                          
checkpoint = tf.train.Checkpoint(module=model)      
checkpoint_path = '/work/long_lab/qli/Enformer_DTL/FromDGX/real_enformer_download/checkpoint/'                    
latest = tf.train.latest_checkpoint(checkpoint_path)
print(latest)
status = checkpoint.restore(latest)


###Evaluate final model on test datasets        
###Read tissue specific tracks
tissue_specific_trakcs_index_df = pd.read_csv("/work/long_lab/qli/Enformer_DTL/FromDGX/41592_2021_1252_MOESM3_ESM_ori_"+tissue+"_chip_tf.csv")
tissue_specific_trakcs_index_df["track_index"] = tissue_specific_trakcs_index_df["track_index"].astype("int")
tissue_specific_trakcs_index = list(tissue_specific_trakcs_index_df["track_index"])
print(tissue_specific_trakcs_index)

@tf.function
def predict(x):
    return model(x, is_training=False)['human']

for i, batch in tqdm(enumerate(human_test_dataset)):
    if i < hyper_paras.evaluate_steps: 
        print("Saving tensor values for "+str(i+1))
        
        true_matrix = np.squeeze(batch['target'].numpy()) #transform from (1, 896, 5313) to (896, 5313)
        breast_tf_true_matrix =true_matrix[:, tissue_specific_trakcs_index]
        
        predict_matrix = np.squeeze(predict(batch['sequence']).numpy()) #transform from (1, 896, 5313) to (896, 5313)
        breast_tf_predict_matrix =predict_matrix[:, tissue_specific_trakcs_index]
        
        np.save(str(i+1)+".ENF.true.npy", breast_tf_true_matrix)
        np.save(str(i+1)+".ENF.predict.npy", breast_tf_predict_matrix)
        print("Finish saving tensor values for "+str(i+1))
