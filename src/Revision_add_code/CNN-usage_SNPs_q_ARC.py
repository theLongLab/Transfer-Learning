#!/usr/bin/env python
# coding: utf-8
# Edit: 20260104

import os
import sys
import time
import numpy as np
import tensorflow as tf
import sonnet as snt
import kipoiseq
from kipoiseq import Interval
from utils import one_hot_encode, FastaStringExtractor
import Baseline_CNN

# -----------------------
# GPU setup
# -----------------------
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpu_index = 0
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[gpu_index], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        print(f"Using GPU: {gpus[gpu_index].name}")
    except (RuntimeError, IndexError) as e:
        print("GPU setup failed, falling back to CPU:", e)
else:
    print("No GPU detected, using CPU")

# -----------------------
# Configuration
# -----------------------
tissue = "breast"
num_outputs = 275
SEQUENCE_LENGTH = 196608
SNPs_per_file = "200K"

file_index = sys.argv[1]

base_dir = "/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX"
snp_file = f"{base_dir}/{tissue}_snps_hg38/{file_index}.{SNPs_per_file}.txt"
out_dir = f"{base_dir}/1_revision_DTL/CNN_{tissue}_hg38_scores"
os.makedirs(out_dir, exist_ok=True)
out_file = f"{out_dir}/{file_index}.{SNPs_per_file}.SNPs.scores.txt"

# -----------------------
# FASTA
# -----------------------
fasta_file = f"{base_dir}/hg38.ml.fa"
fasta_extractor = FastaStringExtractor(fasta_file)
seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
    reference_sequence=fasta_extractor
)

# -----------------------
# Load SNP list
# -----------------------
with open(snp_file) as f:
    snps = [line.strip() for line in f if line.strip()]

# -----------------------
# Model
# -----------------------
model = Baseline_CNN.SimpleCNN(num_outputs=num_outputs)

# Build variables (REQUIRED)
_ = model(tf.zeros([1, SEQUENCE_LENGTH, 4]), is_training=False)

# -----------------------
# Restore checkpoint (model only)
# -----------------------
checkpoint_dir = os.path.join(
    base_dir,
    f"human_tfrecords_{tissue}",
    "checkpoint_CNN"
)

ckpt = tf.train.Checkpoint(model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print(f"Restored from {ckpt_manager.latest_checkpoint}")
else:
    raise RuntimeError("❌ No CNN checkpoint found.")

# -----------------------
# Output header
# -----------------------
with open(out_file, "w") as fw:
    fw.write("SNP_ID," + ",".join([f"track_{i}" for i in range(num_outputs)]) + "\n")
    # -----------------------
    # SNP loop
    # -----------------------
    for i, snp in enumerate(snps, 1):
        start_time = time.time()
        try:
            chr_, pos, ref, alt = snp.split("_")
            pos = int(pos)
            # Skip SNPs too close to chromosome start
            if pos <= 200_000:
                continue
            variant = kipoiseq.Variant(f"chr{chr_}", pos, ref, alt)
            interval = Interval(
                variant.chrom,
                variant.start,
                variant.start
            ).resize(SEQUENCE_LENGTH)
            center = interval.center() - interval.start
            ref_seq = seq_extractor.extract(interval, [], anchor=center)
            alt_seq = seq_extractor.extract(interval, [variant], anchor=center)
            ref_pred = model(
                one_hot_encode(ref_seq)[None],
                is_training=False
            )["human"][0]
            alt_pred = model(
                one_hot_encode(alt_seq)[None],
                is_training=False
            )["human"][0]
            # Center bin (robust)
            center_bin = ref_pred.shape[0] // 2
            diff = alt_pred[center_bin - 1:center_bin + 1] - \
                   ref_pred[center_bin - 1:center_bin + 1]
            diff_mean = diff.numpy().mean(axis=0)
            fw.write(snp + "," + ",".join(map(str, diff_mean)) + "\n")
            if i % 100 == 0:
                fw.flush()
        except Exception as e:
            print(f"[WARNING] SNP {snp} failed: {e}")
            continue
