#!/usr/bin/env python
# coding: utf-8
# Edit: 20240307

# Conda activate enformer_env
# enformer_env             /work/long_lab/qli/anaconda3/envs/enformer_env


####Load packages
import sonnet as snt
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
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
import tfenformer_prostate
import pdb
import sonnet as snt
import wandb
wandb.login()


###Load devices
#device_index = '0' #0/1/2/3 for GPU and blank for CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = device_index
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


####Set parameters
hyper_paras=Hyper_paras()
hyper_paras.batch_size = 8
hyper_paras.transfer_learning = True
hyper_paras.update_transferred_weight = True
hyper_paras.accumulate = False #True
hyper_paras.pre_train_steps = 1 #Not apply optimizer to parameters 
hyper_paras.train_steps = 2560  #34K samples, 64 samples to update weights once, 531 updates for 34K samples. 256*10==2560    [1k, 5k, 10k, 20k, 25k, 30k, 35k]
hyper_paras.evaluate_steps = 512  #??? 1,937, should test all samples in test datasets
tissue="prostate"

####Load datasets
human_dataset = get_dataset('human', 'train', tissue).batch(hyper_paras.batch_size).repeat(hyper_paras.epoch).prefetch(hyper_paras.batch_size)
train_data_it = iter(human_dataset)
human_valid_dataset = get_dataset('human', 'valid', tissue).batch(hyper_paras.batch_size).repeat(hyper_paras.epoch).prefetch(hyper_paras.batch_size)
valid_data_it=iter(human_valid_dataset)
human_test_dataset = get_dataset('human', 'test', tissue).batch(hyper_paras.batch_size).repeat(hyper_paras.epoch).prefetch(hyper_paras.batch_size)
test_data_it=iter(human_test_dataset)

#### Train and Evaluate methods
def create_step_function_pretrain(model):
    @tf.function
    def pretrain_step(batch, head, optimizer_clip_norm_global=0.2):
        with tf.GradientTape() as tape:
            outputs = model(batch['sequence'], is_training=True)[head]
            loss = tf.reduce_mean(tf.keras.losses.poisson(batch['target'], outputs))
        return loss
    return pretrain_step

def create_step_function(model, tissue, optimizer1, optimizer2,batch_size):
    @tf.function
    def train_step(batch, head, optimizer_clip_norm_global=0.2):
        shape_list=get_shape_list("human", tissue, "trainable_variables_shape_list_t10ph.pickle")
        skip_variables_len = 243-len(shape_list) #243-6=237
        with tf.GradientTape() as tape:
            outputs = model(batch['sequence'], is_training=True)[head]
            loss = tf.reduce_mean(tf.keras.losses.poisson(batch['target'], outputs))
        if hyper_paras.update_transferred_weight:
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer1.apply(gradients[:2], model.trainable_variables[:2])
            optimizer2.apply(gradients[(2+skip_variables_len):], model.trainable_variables[(2+skip_variables_len):])
        else:
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer1.apply(gradients[:2], model.trainable_variables[:2])
        return (loss)
    return train_step

@tf.function
def valid_step(model,batch,head):
    outputs=model(batch['sequence'],is_training=False)[head]
    loss=tf.reduce_mean(tf.keras.losses.poisson(batch['target'], outputs))
    return loss

def evaluate_model(model, dataset, head, max_steps=None):
    metric = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
    @tf.function
    def predict(x):
        return model(x, is_training=False)[head]
    for i, batch in tqdm(enumerate(dataset)):
        if max_steps is not None and i > max_steps: ## will look at maximum 100 datasets 
              break
        metric.update_state(batch['target'], predict(batch['sequence']))
    return metric.result()


# # Model Train
run = wandb.init(project="TFEnformer-wandb",config={"learning_rate": 0.00005,
                                                    "train steps":hyper_paras.train_steps,
                                                    "batch_size":hyper_paras.batch_size,
                                                    "loss_function":"Poisson negative log-likelihood loss function",
                                                    "architecture":"Attention",
                                                    "dataset:":"hg38_196k_275raw"})
config=wandb.config


learning_rate = tf.Variable(0.00005, trainable=False, name='learning_rate') ##!!!
optimizer1= snt.optimizers.Adam(learning_rate=learning_rate)
optimizer2= snt.optimizers.Adam(learning_rate=learning_rate)

model = tfenformer_prostate.Enformer(channels=1536,
                          num_heads=8,
                          num_transformer_layers=11,
                          pooling_type='attention')
pretrain_step = create_step_function_pretrain(model)


#### Pretrain the model and transfer weights
for i in tqdm(range(hyper_paras.pre_train_steps)):
    batch_human = next(train_data_it)
    train_loss_human=pretrain_step(batch=batch_human, head='human')
    # End of epoch.
    print('train_loss_human', train_loss_human.numpy())


#### Load previous parameters into current models
checkpoint_path="/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/real_enformer_download/variables/variables"

#### Local model's variables
all_variables = list(model.variables)
all_variables_names_list=[]
for ind,v in enumerate(all_variables):
    all_variables_names_list.append(v.name)

##Google trained model's variables' weights
GEnformer_variables_name_list = tf.train.list_variables(checkpoint_path) ##list all variables names


##Match local variables and Google trained models
variables_files=open("/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/DTL_202211/Enformer_all_Lvariables_matched_withindex_dict_p357.json","r")
variables_match_dict=json.load(variables_files)


##Assign Google Enformer model variables weights to the local model
variables_match_dict_key_list = list(variables_match_dict.keys())
for key in variables_match_dict_key_list[2:]:
    value=variables_match_dict.get(key)
    key_index=int(key.split("-")[0])
    value_index=int(value.split("-")[0])
    all_variables[key_index].assign(tf.train.load_variable(checkpoint_path,GEnformer_variables_name_list[value_index][0]),name=variables_match_dict_key_list[key_index])

# ### Train model with 275 raw TF tracks
train_step = create_step_function(model, tissue, optimizer1, optimizer2, hyper_paras.batch_size)

###If there are prvious check points, restor model from there. 
checkpoint_root = "/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX/human_tfrecords_"+tissue+"/checkpoint"
if not os.path.exists(checkpoint_root):
    os.system('mkdir -p '+checkpoint_root)
checkpoint_name = "t10ph"
save_prefix = os.path.join(checkpoint_root, checkpoint_name)
checkpoint = tf.train.Checkpoint(module=model)
latest = tf.train.latest_checkpoint(checkpoint_root)
if latest is not None:
    checkpoint.restore(latest)

### Training
loss_log=open(checkpoint_root+"/"+checkpoint_name+"_loss_log_human.txt","a")
for global_step in range(hyper_paras.train_steps): 
    batch_human = next(train_data_it)
    train_loss_human=train_step(batch=batch_human, head='human')
    #Every 32 train_steps(8*64=512 samples), we check validaition and test? End of train_steps, calculate validate loss and pearsonR
    if ((global_step+1)%128==0):
        valid_loss_human=[]
        for j in range(hyper_paras.evaluate_steps):
            valid_batch_human=next(valid_data_it)
            valid_loss_human.append(float(valid_step(model, batch=valid_batch_human, head='human')))
        metrics_human = evaluate_model(model,
                           dataset=human_valid_dataset,
                           head='human',
                           max_steps=hyper_paras.evaluate_steps)
        print({k: v.numpy().mean() for k, v in metrics_human.items()})
        test_metrics_human = evaluate_model(model,
                       dataset=human_test_dataset,
                       head='human',
                       max_steps=hyper_paras.evaluate_steps)
        print({k: v.numpy().mean() for k, v in test_metrics_human.items()})
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        loss_log.write("date and time: "+ date_time+", global_step: "+str(global_step)+
        ", train_loss_human: "+str(np.mean(train_loss_human))+', valid_loss_human: '+str(np.mean(valid_loss_human))+
        ", valid pearson R: "+str(np.mean(metrics_human["PearsonR"]))+", test pearson R: "+str(np.mean(test_metrics_human["PearsonR"]))+"\n")
        wandb.log({"global_step": global_step,"train_loss":np.mean(train_loss_human),"valid_loss":np.mean(valid_loss_human),"valid_PearsonR":np.mean(metrics_human["PearsonR"]), "test_PearsonR":np.mean(test_metrics_human["PearsonR"])})
        checkpoint.save(save_prefix)
    
## Finish wandb 
loss_log.close()
run.finish()
