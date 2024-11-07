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

##################For DTL
###Hyperparameters Definition
hyper_paras=Hyper_paras()
hyper_paras.evaluate_steps = 1937  #1,937, should test all samples in test datasets
tissue=sys.argv[1] #breast, prostate
model="DTL" #"DTL"
batch="2024"
output_prefix=tissue+"_"+model+"_"+batch

###Load test datasets
human_test_dataset = get_dataset('human', 'test', tissue).batch(hyper_paras.batch_size).repeat(hyper_paras.epoch).prefetch(hyper_paras.batch_size)

###Prepare DTL model
if tissue=="breast":
    model = tfenformer_breast.Enformer(channels=1536,
                              num_heads=8,
                              num_transformer_layers=11,
                              pooling_type='attention')
elif  tissue=="prostate":
    model = tfenformer_prostate.Enformer(channels=1536,
                              num_heads=8,
                              num_transformer_layers=11,
                              pooling_type='attention')
                          
checkpoint = tf.train.Checkpoint(module=model)      
checkpoint_path = './human_tfrecords_'+tissue+'/checkpoint/'                    
latest = tf.train.latest_checkpoint(checkpoint_path)
print(latest)
status = checkpoint.restore(latest)


###Evaluate final model on test datasets 
if not os.path.exists("./TestSetPearsonR/"+output_prefix+"_true_predict_npy"):
    os.system("mkdir ./TestSetPearsonR/"+output_prefix+"_true_predict_npy")
 
@tf.function
def predict(x):
    return model(x, is_training=False)['human']

for i, batch in tqdm(enumerate(human_test_dataset)):
    if i < hyper_paras.evaluate_steps: 
        print("Saving tensor values for "+str(i+1))
        np.save("./TestSetPearsonR/"+output_prefix+"_true_predict_npy/"+str(i+1)+".true.npy", batch['target'].numpy())
        np.save("./TestSetPearsonR/"+output_prefix+"_true_predict_npy/"+str(i+1)+".predict.npy", predict(batch['sequence']).numpy())
        print("Finish saving tensor values for "+str(i+1))
        
