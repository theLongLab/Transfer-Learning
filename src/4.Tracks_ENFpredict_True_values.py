#!/usr/bin/env python
# coding: utf-8

'''
Initilize 20230630
Usage: Calculate person R for each track across all position in test sets
Last check: 20240709
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
import tfenformer_lung
import tfenformer_prostate
import enformer
import sys

assert snt.__version__.startswith('2.0')
print(tf.__version__)
        
##################For ENF
###Hyperparameters Definition
hyper_paras=Hyper_paras()
hyper_paras.evaluate_steps = 1937  #1,937, should test all samples in test datasets
tissue=sys.argv[1] #breast, prostate
model="ENF" #"DTL"
batch="8"
output_prefix=tissue+"_"+model+"_"+batch

def tfrecord_files(organism, subset):
  # Sort the values by int(*).
  return sorted(tf.io.gfile.glob(os.path.join(
      '/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/sonnet_enformer_download/Enformer2021human/enformer_393k_tfrecords/', f'{subset}-*.tfr'
  )), key=lambda x: int(x.split('-')[-1].split('.')[0]))


###Load test datasets
def get_dataset(organism, subset, tissue, num_threads=8):
    metadata = get_metadata()
    dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset),
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads)
    dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                        num_parallel_calls=num_threads)
    return dataset
    
def get_metadata():
    # Keys:
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
    path = os.path.join('/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/sonnet_enformer_download/Enformer2021human/enformer_393k_tfrecords/', 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)

human_test_dataset = get_dataset('human', 'test', tissue).batch(hyper_paras.batch_size).repeat(hyper_paras.epoch).prefetch(hyper_paras.batch_size)

###Prepare DTL model
model = enformer.Enformer(channels=1536,
                          num_heads=8,
                          num_transformer_layers=11,
                          pooling_type='attention')
                          
checkpoint = tf.train.Checkpoint(module=model)      
checkpoint_path = '/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/sonnet_enformer_download/'                    
latest = tf.train.latest_checkpoint(checkpoint_path)
print(latest)
status = checkpoint.restore(latest)


###Evaluate final model on test datasets 
if not os.path.exists("/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/TestSetPearsonR/"+output_prefix+"_true_predict_npy"):
    os.system("mkdir /work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/TestSetPearsonR/"+output_prefix+"_true_predict_npy")
 
@tf.function
def predict(x):
    return model(x, is_training=False)['human']

columns_to_extract=[]
if tissue=="breast":
    columns_to_extract=[2373, 2383, 2384, 2507, 2666, 2775, 3003, 3087, 3487, 3601, 4052, 4335]
elif tissue=="prostate":
    columns_to_extract=[1836, 2060, 2681, 2812, 3678, 4014, 4270]

for i, batch in tqdm(enumerate(human_test_dataset)):
    if i < hyper_paras.evaluate_steps: 
        print("Saving tensor values for "+str(i+1))
        np.save("/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/TestSetPearsonR/"+output_prefix+"_true_predict_npy/"+str(i+1)+".true.npy", batch['target'].numpy())
        np.save("/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/TestSetPearsonR/"+output_prefix+"_true_predict_npy/"+str(i+1)+".predict.npy", predict(batch['sequence']).numpy())
        print("Finish saving tensor values for "+str(i+1))

