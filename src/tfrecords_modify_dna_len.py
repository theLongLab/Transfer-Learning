#!/usr/bin/env python
# Copyright 2019 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser
import os
import sys
import numpy as np
import pdb
import pysam
import json
import functools
from basenji.bin.basenji_data import ModelSeq
from basenji.basenji.dna_io import dna_1hot, dna_1hot_index
import tensorflow as tf
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]='false'


"""
basenji_data_write.py

Write TF Records for batches of model sequences.

Notes:
-I think target_start and target_end are remnants of my previous data2 pipeline.
 If I see this again beyond 8/2020, remove it.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <fasta_file> <seqs_bed_file> <tfr_file>'
  parser = OptionParser(usage)
  parser.add_option('-s', dest='start_i',
      default=0, type='int',
      help='Sequence start index [Default: %default]')
  parser.add_option('-e', dest='end_i',
      default=None, type='int',
      help='Sequence end index [Default: %default]')
  parser.add_option('--te', dest='target_extend',
      default=None, type='int', help='Extend targets vector [Default: %default]')
  parser.add_option('-u', dest='umap_npy',
      help='Unmappable array numpy file')
  parser.add_option('--umap_clip', dest='umap_clip',
      default=1, type='float',
      help='Clip values at unmappable positions to distribution quantiles, eg 0.25. [Default: %default]')
  parser.add_option('--umap_tfr', dest='umap_tfr',
      default=False, action='store_true',
      help='Save umap array into TFRecords [Default: %default]')
  parser.add_option('-x', dest='extend_bp',
      default=0, type='int',
      help='Extend sequences on each side [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide input arguments.')
  else:
    fasta_file = args[0]
    seqs_bed_file = args[1]
    tfr_file = args[2]
    
'''
fasta_file='hg38.ml.fa'
seqs_bed_file='sequences_hg38_all_196k.bed'
tfr_file='test-0-7.tfr'
'''  

  ################################################################
  # read model sequences

  model_seqs = []
  for line in open(seqs_bed_file):
    a = line.split()
    model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2]),None))

  if options.end_i is None:
    options.end_i = len(model_seqs)

  num_seqs = options.end_i - options.start_i

  ################################################################
  # Load tfr from original tfr
  
  dataset = get_dataset('human', 'test', tfr_file).batch(1)
  num_elements = count_elements(dataset)
  targets = collect_targets(dataset, num_elements)


  ################################################################
  # write TFRecords

  # open FASTA
  fasta_open = pysam.Fastafile(fasta_file)

  # define options
  tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')
  
  data_iter = iter(dataset)
  with tf.io.TFRecordWriter('./enformer_393k_tfrecords/'+tfr_file, tf_opts) as writer:
    for si in range(num_seqs):
      msi = options.start_i + si
      mseq = model_seqs[msi]
      mseq_start = mseq.start - options.extend_bp
      mseq_end = mseq.end + options.extend_bp
      # read FASTA
      # seq_dna = fasta_open.fetch(mseq.chr, mseq.start, mseq.end)
      seq_dna = fetch_dna(fasta_open, mseq.chr, mseq_start, mseq_end)
      # one hot code (N's as zero)
      seq_1hot = dna_1hot(seq_dna, n_uniform=False, n_sample=False)
      # seq_1hot = dna_1hot_index(seq_dna) # more efficient, but fighting inertia
      # hash to bytes
      data=next(data_iter)
      target = data['target'].numpy().reshape(1, 896, 5313).astype('float16')  
      #Note float16 is necessary. The default float32 will lead to shape error!One day to debug that.
      features_dict = {
        'sequence': feature_bytes(seq_1hot),
        'target': feature_bytes(target)
        }
      # add unmappability
      if options.umap_tfr:
        features_dict['umap'] = feature_bytes(unmap_mask[msi,:])
      # write example
      example = tf.train.Example(features=tf.train.Features(feature=features_dict))
      writer.write(example.SerializeToString())

    fasta_open.close()

# Example usage: creating a dataset from TFRecord files

def count_elements(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count
    
def collect_targets(dataset, num_elements):
    targets = np.empty((num_elements, 896, 5313), dtype=np.float32)
    for i, data in enumerate(dataset):
        if i >= num_elements:
            break
        target = data['target'].numpy().squeeze(axis=0)  # Remove the leading dimension of size 1
        targets[i, :, :] = target
    return targets

def get_dataset(organism, subset, tfr_file, num_threads=8):
    metadata = get_metadata()
    dataset = tf.data.TFRecordDataset('./basenji_132k_tfrecords/'+tfr_file,
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads)
    dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                        num_parallel_calls=num_threads)
    return dataset

@tf.autograph.experimental.do_not_convert
def deserialize(serialized_example, metadata):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)
    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                      (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)
    return {'sequence': sequence,
          'target': target}

def get_metadata():
    # Keys:
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
    path = os.path.join('/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/sonnet_enformer_download/Enformer2021human/basenji_132k_tfrecords/', 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)


def fetch_dna(fasta_open, chrm, start, end):
  """Fetch DNA when start/end may reach beyond chromosomes."""
  # initialize sequence
  seq_len = end - start
  seq_dna = ''
  # add N's for left over reach
  if start < 0:
    seq_dna = 'N'*(-start)
    start = 0
  # get dna
  seq_dna += fasta_open.fetch(chrm, start, end)
  # add N's for right over reach
  if len(seq_dna) < seq_len:
    seq_dna += 'N'*(seq_len-len(seq_dna))
  return seq_dna


def feature_bytes(values):
  """Convert numpy arrays to bytes features."""
  values = values.flatten().tobytes()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def feature_floats(values):
  """Convert numpy arrays to floats features.
     Requires more space than bytes."""
  values = values.flatten().tolist()
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
