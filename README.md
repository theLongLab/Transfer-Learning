# Users’ Manual of Transfer-Learning
## Overview

![My Image](/Figure/Figure1_20240709.png)
**The overview of the transfer learning workflow and construction of tCRAs.** A) A schematic of general TL: a small number of dedicated data may re-train a selected subset of nodes to redirect an existing model. B) Enformer model was trained on input DNA sequences (196kb) to predict multiple human epigenomic tracks (5,313) at 128-bp resolution. The network is composed of four blocks (indicated by each rectangle) from convolutional layers, transformer layers, pointwise convolution, and output heads convolution. The output shapes of these blocks are given by tuples on the bottom-right corner, in which C (=1,536) indicates the number of channels in CNN. The number of trainable parameters for each block is listed on the bottom-left corner. C) Our TL used majority of existing Enformer architecture together with its trained parameters by retaining its input and first three blocks. The only layer undergoes re-training is the output heads convolution layers, tailed to target-tissue epigenetic datasets (i.e., 275 tracks of TFs ChIP-seq for breast cancer). D) Left panel: An illustration of tCRAs for a specific genetic variant estimated by calculating the differences of predicted regulatory activity value between reference allele (A) and alternative allele (C) of a variant. Right panel: based on TL outcomes, we generated an activity score for each genetic variant for each track, which can be utilized in downstream analyses including association study.

## Prerequisites
- **Required python package**
  - Python 3.8.12 and above
  - dm-sonnet (2.0.0)
  - kipoiseq (0.5.2)
  - numpy (1.19.5)
  - pandas (1.2.3)
  - tensoflow (2.4.1)

## Download Enformer model and variants scores from Enformer models
- The final Enformer model is saved in checkpoint format and is available for download at: gs://dm-enformer/models/enformer/sonnet_weights/. Please download the files and place them in ./data/Enformer_checkpoint/.
- Variant effect scores for all common variants (MAF > 0.5% in any population) included in the 1000 Genomes Project from Enformer model are available:
  https://github.com/google-deepmind/deepmind-research/tree/master/enformer

## Construction of TensorFlow records containing both DNA sequences and target ChIP-seq profiles
1. DNA sequences to format train/valide/test can be found through this file: sequences_hg38_all_196k.bed. Huamn genome fasta file (GRCh38) and its index can be downloaded from: http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz.
2. TF ChIP-seq bigwig file could be downloaded from the Cistrom website: http://cistrome.org/
3. Code to generate tensorflow records are available below. For more details, please visit: https://github.com/calico/basenji
```
python basenji_data_read.py --crop 40960 -w 128 -u mean -c 32.00 -s 2.00 ./data/49535.bw ./data/sequences_hg38_all_196k.bed seqs_cov/110.h5
python basenji_data_write.py -s 0 -e 256 --umap_clip 1.000000 hg38.ml.fa sequences_hg38_all_196k.bed seqs_cov/ tfrecords/train-0-0.tfr
```
## Conduct transfer Learning
```
python 1.TL_TF_t10ph_breast.py
python 1.TL_TF_t10ph_prostate.py
```
## Calculate tCRAs scores from TL-breast and TL-prostate
GWAS summary statistics are required to generate input files for tCRA scores. Example input files for calculating tCRA scores can be found at ./data/Example_input_tCRAs.txt. Variants are in format of chr_pos_ref_alt.
```
python 2.TL_tCRAs_scores.py
```
## Conduct TWAS analyses
We utilized gene expression and genotype downloaded from GTEx (https://gtexportal.org/home/datasets).

In the TWAS process, a predictive model is built by combining gene expression data (e.g., GTEx expression data normalized with PEER factors) and cis-variant genotypes within a specific distance around the gene TSS (e.g., ±1Mb). Using elastic-net regression, a predictive model for each gene is trained to estimate the effect of each variant on gene expression. Elastic-net combines L1 and L2 penalties to optimize weights, which capture each variant’s influence on gene expression.
```
Rscript TWAS_modelbuilding_GTEx.R
Rscript WeightsSummaryToDB.R
```

In the second step, the trained weights are applied to GWAS summary statistics to calculate association Z-scores for each gene. These Z-scores estimate the association between predicted gene expression and disease risk, with significance evaluated using P-values after Bonferroni correction. This approach ultimately allows researchers to identify genes whose predicted expression levels are statistically associated with disease risk, providing insight into the genetic architecture of complex traits.
```
SPrediXcan.py --model_db_path breast_2024_TL_1500K.db --covariance breast_2024_TL_1500K_cov.txt.gz --model_db_snp_key rsid --gwas_folder ./Cancer_GWAS_SS/breast_assoc_dosage/ --gwas_file_pattern ".*gz" --snp_column SNP --effect_allele_column A1 --non_effect_allele_column A2 --beta_column BETA  --pvalue_column P --output_file  breast_2024_TL_1500K.TWAS --verbosity 2
```
More information can be found in https://github.com/theLongLab/TF-TWAS

## Contacts
Qing Li: liqingbioinfo@gmail.com  
Quan Long: quan.long@ucalgary.ca

## Copyright License (MIT Open Source)
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
