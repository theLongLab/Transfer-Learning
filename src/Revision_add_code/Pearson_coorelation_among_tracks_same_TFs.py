#!/usr/bin/env python
# coding: utf-8

import os
import sys
import itertools

import numpy as np
import pandas as pd
import sonnet as snt
from tqdm import tqdm
import matplotlib.pyplot as plt


### Define functions
def tfrecord_files(organism, subset, tissue):
  # Sort the values by int(*).
  return sorted(tf.io.gfile.glob(os.path.join(
      f'//work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/human_tfrecords_{tissue}/tfrecords/', f'{subset}-*.tfr'
  )), key=lambda x: int(x.split('-')[-1].split('.')[0]))

def get_metadata(tissue):
    # Keys:
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
    path = os.path.join(f'//work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/human_tfrecords_{tissue}/', 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)

def get_dataset(organism, subset, tissue, num_threads=1):
    metadata = get_metadata(tissue)
    dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset, tissue),
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads)
    dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                        num_parallel_calls=num_threads)
    return dataset


def paired_pearson_r_with_pairs(pred, track_indices):
    """
    pred: [B, T, n_tracks]
    Returns:
      dict {(track_i, track_j): r}
    """
    pred_sel = tf.gather(pred, track_indices, axis=-1)   # [B, T, k]
    pred_flat = tf.reshape(pred_sel, [-1, pred_sel.shape[-1]])
    pred_flat = tf.cast(pred_flat, tf.float32)
    mean = tf.reduce_mean(pred_flat, axis=0, keepdims=True)
    centered = pred_flat - mean
    cov = tf.matmul(centered, centered, transpose_a=True)
    var = tf.linalg.diag_part(cov)
    std = tf.sqrt(var)
    corr = cov / (tf.tensordot(std, std, axes=0) + 1e-8)
    corr = corr.numpy()
    pair_corrs = {}
    for i, j in itertools.combinations(range(len(track_indices)), 2):
        pair_corrs[(track_indices[i], track_indices[j])] = corr[i, j]
    return pair_corrs


def prepare_tissue_data(tissue, df_tfs, batch_size=None):
    # Hyper parameters
    hyper_paras = Hyper_paras()
    if batch_size is not None:
        hyper_paras.batch_size = batch_size
    # Dataset
    human_test_dataset = (get_dataset("human", "test", tissue).batch(hyper_paras.batch_size).prefetch(tf.data.AUTOTUNE))
    # TF → track indices
    track_dict = (df_tfs.groupby("Factor")["Index"].apply(list).to_dict())
    # Track index → cell line
    index_to_cell = (df_tfs.set_index("Index")["Cell_line"].to_dict())
    return human_test_dataset, track_dict, index_to_cell

def compute_pairwise_track_correlations(dataset, track_dict, index_to_cell, tissue, out_csv=None ):
    pairwise_corrs = defaultdict(list)
    # Iterate through dataset
    for batch_id, batch in tqdm(enumerate(dataset)):
        x = batch["target"]   # [B, T, n_tracks]
        for tf_name, track_idxs in track_dict.items():
            pair_corr_dict = paired_pearson_r_with_pairs(x, track_idxs)
            for (ti, tj), r in pair_corr_dict.items():
                if np.isnan(r):
                    continue
                ci = index_to_cell.get(ti)
                cj = index_to_cell.get(tj)
                pairwise_corrs[(tf_name, ti, tj, ci, cj)].append(r)
    # Aggregate across samples
    results = []
    for (tf_name, ti, tj, ci, cj), r_list in pairwise_corrs.items():
        results.append({
            "TF": tf_name,
            "track_i": ti,
            "cell_i": ci,
            "track_j": tj,
            "cell_j": cj,
            "mean_pairwise_corr": np.mean(r_list),
            "n_samples": len(r_list)
        })

    df = pd.DataFrame(results)
    # Save (optional)
    if out_csv is None:
        out_csv = f"Target_tack_pairwise_corr_testset_{tissue}.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return df

def plot_pairwise_corr_by_tf_and_cell(df, tissue, y_col="mean_pairwise_corr", tf_col="TF", cell_i_col="cell_i", cell_j_col="cell_j",figsize_scale=0.7):
    # TF order by overall median
    order = (df.groupby(tf_col)[y_col].median().sort_values(ascending=False).index)
    # Prepare dataframe for Panel B
    df_plot = df.copy()
    df_plot["cell_pair"] = np.where(
        df_plot[cell_i_col] == df_plot[cell_j_col],
        "Same cell line",
        "Different cell line"
    )
    # Create figure
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(max(12, figsize_scale * len(order)), 5),sharey=True)
    # Panel A: All pairs
    palette_A = sns.color_palette("husl", 16)
    sns.boxplot(data=df,x=tf_col,y=y_col,order=order,palette=palette_A,showfliers=False,ax=axes[0])
    axes[0].set_title("A  All track pairs", loc="left", fontweight="bold")
    axes[0].set_xlabel("Transcription Factor")
    axes[0].set_ylabel("Mean track–track Pearson r\n(across 1,937 test sequences)")
    axes[0].tick_params(axis="x", rotation=45)
    # Panel B: Same vs different cell
    palette_B = {
        "Same cell line": "white",
        "Different cell line": "grey"
    }
    sns.boxplot(data=df_plot,x=tf_col,y=y_col,hue="cell_pair",hue_order=["Same cell line", "Different cell line"], order=order,palette=palette_B,showfliers=False,ax=axes[1])
    axes[1].set_title("B  Stratified by cell line", loc="left", fontweight="bold")
    axes[1].set_xlabel("Transcription Factor")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(title="", loc="best")
    # Final
    fig.suptitle(tissue.capitalize(), y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"Target_tack_pairwise_corr_testset_{tissue}.pdf", dpi=300)


#### Track-track correlation across 1937 test datasets
breast_tfs=pd.read_excel("/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/1_revision_DTL/SupplementaryTables_20251223.xlsx", sheet_name="S1", header=1)
tissue = "breast"
df_tfs=breast_tfs
human_test_dataset, track_dict, index_to_cell = prepare_tissue_data(tissue, df_tfs)
track_pairwise_df = compute_pairwise_track_correlations(human_test_dataset, track_dict, index_to_cell, tissue)
plot_pairwise_corr_by_tf_and_cell(track_pairwise_df, tissue)


#### Track-track correlation across 1937 test datasets
prostate_tfs=pd.read_excel("/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/1_revision_DTL/SupplementaryTables_20251223.xlsx", sheet_name="S2", header=1)
tissue = "prostate"
df_tfs=prostate_tfs
human_test_dataset, track_dict, index_to_cell = prepare_tissue_data(tissue, df_tfs)
track_pairwise_df = compute_pairwise_track_correlations(human_test_dataset, track_dict, index_to_cell, tissue)
plot_pairwise_corr_by_tf_and_cell(track_pairwise_df, tissue)

