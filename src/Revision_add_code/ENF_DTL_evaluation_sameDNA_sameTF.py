#!/usr/bin/env python
# coding: utf-8
# Edit: 20251223

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
import tfenformer_prostate
import enformer
import sys
assert snt.__version__.startswith('2.0')
print(tf.__version__)


# ### Enformer
def tfrecord_files(organism, subset):
  # Sort the values by int(*).
  return sorted(tf.io.gfile.glob(os.path.join(
      '//work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/1_revision_DTL/DTL_ENF_comparison/ENF_checkpoint/test_tfr/', f'{subset}-*.tfr'
  )), key=lambda x: int(x.split('-')[-1].split('.')[0]))

def get_dataset(organism, subset, tissue, num_threads=1):
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
    path = os.path.join('/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/1_revision_DTL/DTL_ENF_comparison/ENF_checkpoint/test_tfr/', 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)
    
def average_tracks(pred, track_indices):
    """
    pred: np.array or tf.Tensor [B, T, C]
    track_indices: list of int
    return: np.array [B, T]
    """
    pred_sel = pred[..., track_indices]      # [B, T, n_tracks]
    pred_mean = np.mean(pred_sel, axis=-1)   # [B, T]
    return pred_mean

def pearson_corr(a, b):
    """
    a, b: np.array [T]
    """
    a = a - a.mean()
    b = b - b.mean()
    return np.sum(a * b) / np.sqrt(np.sum(a*a) * np.sum(b*b))


###ENF moderl
model = enformer.Enformer(channels=1536,
                          num_heads=8,
                          num_transformer_layers=11,
                          pooling_type='attention')
                          
checkpoint = tf.train.Checkpoint(module=model)      
checkpoint_path = '/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/1_revision_DTL/DTL_ENF_comparison/ENF_checkpoint/'                    
latest = tf.train.latest_checkpoint(checkpoint_path)
print(latest)
status = checkpoint.restore(latest)

@tf.function
def predict_enf(x):
    outputs = model(x, is_training=False)
    return outputs['human']   # [B, T, C_enf]


### DTL_prostate model
tissue='prostate'
DTL_model_prostate = tfenformer_prostate.Enformer(channels=1536,
                          num_heads=8,
                          num_transformer_layers=11,
                          pooling_type='attention')

# Restore from the last checkpoint
checkpoint = tf.train.Checkpoint(module=DTL_model_prostate)      
checkpoint_path = f'/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/1_revision_DTL/DTL_ENF_comparison/DTL_{tissue}_checkpoint/'                    
latest = tf.train.latest_checkpoint(checkpoint_path)
print(latest)
status = checkpoint.restore(latest)

@tf.function
def predict_dtl(x):
    outputs = DTL_model_prostate(x, is_training=False)
    return outputs['human']   # [B, T, C_dtl]

### Compare the ENF output and DTL output on the same TF (average across tracks for the same TF) for the same input DNA sequence
hyper_paras=Hyper_paras()
hyper_paras.evaluate_steps = 1937  #1,937, should test all samples in test datasets

human_test_dataset = get_dataset('human', 'test', tissue).batch(hyper_paras.batch_size).repeat(hyper_paras.epoch).prefetch(hyper_paras.batch_size)
test_data_it=iter(human_test_dataset)

Enformer_tissue_tracks_dict = {'POLR2A': [1836, 2681, 2812, 4270]}
DTL_tissue_tracks_dict = {'POLR2A': list(range(289, 331))}

tf_name = 'POLR2A'
rs = []
results = []   # list of dicts

for i, batch in tqdm(enumerate(human_test_dataset)):
    if i >= hyper_paras.evaluate_steps:
        break
    x = batch['sequence']
    # forward
    enf_pred = predict_enf(x).numpy()    # [(1, 896, 5313)]
    dtl_pred = predict_dtl(x).numpy()    # [(1, 896, 357)]
    
    # average across tracks
    enf_avg = average_tracks(enf_pred, Enformer_tissue_tracks_dict[tf_name])  # [B, T]
    dtl_avg = average_tracks(dtl_pred, DTL_tissue_tracks_dict[tf_name])  # [B, T]

    # correlation per sample
    for b in range(enf_avg.shape[0]):
        r = pearson_corr(enf_avg[b], dtl_avg[b])
        rs.append(r)

    print(f"Sequence:{i+1}, {tf_name} mean Pearson r:{np.mean(rs)}")
    results.append({"TF": tf_name,"sequence_id": i + 1,"pearson_r": rs})

### Output final results
df = pd.DataFrame(results)
out_csv = "ENF_vs_DTL_Pearson_summary.csv"
df.to_csv(out_csv, index=False)
print("Saved results to:", out_csv)

