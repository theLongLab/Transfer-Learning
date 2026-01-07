#!/usr/bin/env python
# coding: utf-8
# Edit: 20251230

import os
import numpy as np
import tensorflow as tf
import sonnet as snt
from tqdm import tqdm
import wandb
from utils import *
import Baseline_CNN

# -----------------------
# Environment
# -----------------------
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"] = "true"

devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(devices[0], "GPU")

print(tf.__version__)
assert snt.__version__.startswith("2.")

# -----------------------
# Hyperparameters
# -----------------------
hyper_paras = Hyper_paras()
hyper_paras.batch_size = 32
hyper_paras.accumulation_steps = 2
hyper_paras.train_steps = 340210 #34021 training, 2213 validation, and 1937 test samples 
hyper_paras.evaluate_steps = 60 #batch_size * evaluate_steps
tissue = "breast"
wandb_log_interval_steps = 4

# -----------------------
# Datasets
# -----------------------
train_ds = get_dataset("human", "train", tissue).batch(
    hyper_paras.batch_size).repeat().prefetch(2)
valid_ds = get_dataset("human", "valid", tissue).batch(
    hyper_paras.batch_size).repeat().prefetch(2)
test_ds = get_dataset("human", "test", tissue).batch(
    hyper_paras.batch_size).repeat().prefetch(2)

train_it = iter(train_ds)
valid_it = iter(valid_ds)
test_it = iter(test_ds)

# -----------------------
# Model & Optimizer
# -----------------------
model = Baseline_CNN.SimpleCNN()

# ---- build variables (REQUIRED for Sonnet) ----
dummy_x = tf.zeros([1, 196608, 4])
out = model(dummy_x, is_training=False)["human"]
#print(out.shape)
optimizer = tf.keras.optimizers.Adam(1e-4)

# -----------------------
# Loss
# -----------------------
def compute_loss(pred, target):
    return tf.reduce_mean(
        tf.keras.losses.poisson(target, pred)
    )
    
# -----------------------
# Train step (grad accumulation)
# -----------------------
accum_step = tf.Variable(0, tf.int32)
update_count = tf.Variable(0, tf.int32)

accum_grads = [
    tf.Variable(tf.zeros_like(v), trainable=False)
    for v in model.trainable_variables
]

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        pred = model(batch["sequence"], is_training=True)["human"]
        loss = compute_loss(pred, batch["target"])
    grads = tape.gradient(loss, model.trainable_variables)
    for g_acc, g in zip(accum_grads, grads):
        g_acc.assign_add(g)
    accum_step.assign_add(1)
    def apply_grads():
        optimizer.apply_gradients(
            zip(
                [g / tf.cast(hyper_paras.accumulation_steps, tf.float32)
                 for g in accum_grads],
                model.trainable_variables
            )
        )
        for g_acc in accum_grads:
            g_acc.assign(tf.zeros_like(g_acc))
        accum_step.assign(0)
        update_count.assign_add(1)
        return 0
    tf.cond(
        tf.equal(accum_step, hyper_paras.accumulation_steps),
        apply_grads,
        lambda: 0
    )
    return loss

# -----------------------
# Validation metric
# -----------------------
def evaluate_model(model, dataset, max_steps):
    metric = PearsonR(reduce_axis=(0, 1))
    for i, batch in enumerate(dataset):
        if i >= max_steps:
            break
        pred = model(batch["sequence"], is_training=False)["human"]
        metric.update_state(batch["target"], pred)
    return metric.result()

# -----------------------
# WandB
# -----------------------
wandb.init(
    project="TL_revision",
    name=f"CNN_bs{hyper_paras.batch_size}_acc{hyper_paras.accumulation_steps}_{tissue}",
    config={
        "batch_size": hyper_paras.batch_size,
        "accumulation_steps": hyper_paras.accumulation_steps,
        "train_steps": hyper_paras.train_steps,
        "binning": "128bp → 896 bins",
        "architecture": "Simple CNN"
    }
)

# -----------------------
# Checkpoint
# -----------------------
checkpoint_name = "CNN"
checkpoint_dir = os.path.join(
    "/work/long_lab/qli/Enformer_DTL/ModelTraining_OldDGX",
    f"human_tfrecords_{tissue}",
    f"checkpoint_{checkpoint_name}"
)
os.makedirs(checkpoint_dir, exist_ok=True)
ckpt = tf.train.Checkpoint(model=model,optimizer=optimizer,update_count=update_count)
ckpt_manager = tf.train.CheckpointManager(ckpt,checkpoint_dir,max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print(f"Restored from {ckpt_manager.latest_checkpoint}")
else:
    print("No checkpoint found. Training from scratch.")

# -----------------------
# Two-phase stopping control
# -----------------------
pearson_threshold = 0.35
early_stop_activated = False
early_stop_counter = 0
early_stop_threshold = 0.001
early_stop_patience = 5

best_valid_pr = -np.inf
for step in range(hyper_paras.train_steps):
    # -------- TRAIN --------
    train_loss = train_step(next(train_it))
    # -------- PERIODIC EVALUATION --------
    if (step + 1) % wandb_log_interval_steps == 0:
        # ---- Validation loss (iterator-based) ----
        valid_losses = []
        for _ in range(hyper_paras.evaluate_steps):
            vb = next(valid_it)
            pred = model(vb["sequence"], is_training=False)["human"]
            valid_losses.append(
                compute_loss(pred, vb["target"]).numpy()
            )
        valid_loss = float(np.mean(valid_losses))
        # ---- Validation PearsonR ----
        valid_pr_vec = evaluate_model(
            model,
            valid_ds,
            hyper_paras.evaluate_steps
        )
        valid_pr = float(tf.reduce_mean(valid_pr_vec).numpy())
        # ---- Test PearsonR ----
        test_pr_vec = evaluate_model(
            model,
            test_ds,
            hyper_paras.evaluate_steps
        )
        test_pr = float(tf.reduce_mean(test_pr_vec).numpy())
        # ---- Progress accounting ----
        weight_updates = int(update_count.numpy())
        samples_seen = (
            weight_updates
            * hyper_paras.batch_size
            * hyper_paras.accumulation_steps
        )
        # ---- Logging ----
        wandb.log(
            {
                "train_loss": float(train_loss),
                "valid_loss": valid_loss,
                "valid_PearsonR": valid_pr,
                "test_PearsonR": test_pr,
                "samples_seen": samples_seen,
                "weight_updates": weight_updates,
            },
            step=weight_updates
        )
        print(
            f"[updates={weight_updates}] "
            f"samples_seen={samples_seen}, "
            f"valid_loss={valid_loss:.6f}, "
            f"valid_PearsonR={valid_pr:.4f}, "
            f"test_PearsonR={test_pr:.4f}"
        )
        # ---- Checkpointing (best model) ----
        if valid_pr > best_valid_pr:
            best_valid_pr = valid_pr
            ckpt_manager.save(checkpoint_number=update_count.numpy())
            print(f"\n Saved checkpoint (best valid_PearsonR={best_valid_pr:.4f})")
        # -------------------------------------------------
        # Phase 1 → Phase 2 transition (PearsonR threshold)
        # -------------------------------------------------
        if (not early_stop_activated) and (valid_pr >= pearson_threshold):
            early_stop_activated = True
            early_stop_counter = 0
            print(
                f"\n PearsonR threshold reached: valid_PearsonR={valid_pr:.4f} ≥ {pearson_threshold}"
                "\n Early stopping is now ACTIVATED.\n"
            )
        # -----------------------
        # Early stopping (Phase 2 only)
        # -----------------------
        if early_stop_activated:
            if valid_loss < early_stop_threshold:
                early_stop_counter += 1
                print(
                    f"Early-stop counter: {early_stop_counter}/{early_stop_patience} "
                    f"(valid_loss={valid_loss:.6f})"
                )
            else:
                early_stop_counter = 0
            if early_stop_counter >= early_stop_patience:
                print(
                    f"\n Early stopping triggered AFTER PearsonR ≥ {pearson_threshold}:\n"
                    f"validation loss < {early_stop_threshold} "
                    f"for {early_stop_patience} consecutive checks."
                )
                break

# -----------------------
# Finish
# -----------------------
wandb.finish()