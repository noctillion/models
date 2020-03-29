#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Modifications Copyright 2020 Patrick Hüther (patrick.huether@gmi.oeaw.ac.at)
# - this is a modified version of local_test.sh that was used on a slurm cluster

# This is optimized to run on 4 nodes with 4 V100 each

cd ../deeplab

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

DATASET=${1}

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Set up the working directories.
ROSETTE_FOLDER="${DATASET}"
EXP_FOLDER="exp/train_on_xception65_imagenet"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/${EXP_FOLDER}/export"
PROFILE_DIR="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/${EXP_FOLDER}/profile"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"
mkdir -p "${PROFILE_DIR}"

# Copy locally the Imagenet trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_CKPT="deeplabv3_xception_2018_01_04.tar.gz"

cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

ROSETTE_DATASET="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/tfrecord"

# Train 75000 iterations.
NUM_ITERATIONS=75000
python "${WORK_DIR}"/train.py \
  --dataset="${DATASET}" \
  --logtostderr \
  --train_split="train" \
  --model_variant="xception_65" \
  --decoder_output_stride=4 \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --train_batch_size=128 \
  --train_crop_size="321,321" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/xception/model.ckpt" \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --hard_example_mining_step=100000 \
  --top_k_percent_pixels=0.25 \
  --num_ps_tasks=2 \
  --num_replicas=4 \
  --slow_start_burnin_type='linear' \
  --learning_policy='poly' \
  --slow_start_step=2000 \
  --slow_start_learning_rate=0 \
  --num_clones=4 \
  --base_learning_rate=0.1 \
  --save_summaries_images=true \
  --initialize_last_layer=false \
  --dataset_dir="${ROSETTE_DATASET}"
