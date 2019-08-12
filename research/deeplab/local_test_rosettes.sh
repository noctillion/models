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
#
# This script is used to run local test on ROSETTE dataset. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
python "${WORK_DIR}"/model_test.py -v

# Go to datasets folder and download ROSETTE segmentation dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
#sh download_and_convert_brat.sh

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
ROSETTE_FOLDER="ara_rosetteSet"
EXP_FOLDER="exp/train_on_trainval"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
#TF_INIT_CKPT="deeplabv3_xception_2018_01_04.tar.gz"

cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

ROSETTE_DATASET="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/tfrecord"

# Train 10 iterations.
NUM_ITERATIONS=200000
python "${WORK_DIR}"/train.py \
  --dataset="ara_rosettes" \
  --logtostderr \
  --train_split="train" \
  --model_variant="xception_65" \
  --decoder_output_stride=4 \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --train_crop_size="537,561" \
  --train_batch_size=24 \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --num_clones=4 \
  --save_summaries_images=true \
  --initialize_last_layer=false \
  --last_layer_contain_logits_only=true \
  --dataset_dir="${ROSETTE_DATASET}"

#  --base_learning_rate=0.0004 \
#  --num_clones=1 \
#  --atrous_rates=6 \
#  --atrous_rates=12 \
#  --atrous_rates=18 \
#  --output_stride=16 \

#  --resize_factor=16 \
#  --min_resize_value=320 \
#  --max_resize_value=561 \

#  --base_learning_rate=0.0000005 \
#  --initialize_last_layer=false \
#  --last_layer_contain_logits_only=true \
#  --tf_initial_checkpoint="${INIT_FOLDER}/xception/model.ckpt" \
#  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
#  --weight_decay=1.0 \

#export CUDA_VISIBLE_DEVICES=0

# Run evaluation. This performs eval over the full val split (998 images) and
# will take a while.
python "${WORK_DIR}"/eval.py \
  --dataset="ara_rosettes" \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=12 \
  --atrous_rates=24 \
  --atrous_rates=36 \
  --output_stride=8 \
  --decoder_output_stride=4 \
  --eval_crop_size="537,561" \
  --add_flipped_images=True \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${ROSETTE_DATASET}" \
  --max_number_of_evaluations=1

#  --eval_scales=1.0 \
#  --labels_offset=1 \

# Visualize the results.
python "${WORK_DIR}"/vis.py \
  --dataset="ara_rosettes" \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=12 \
  --atrous_rates=24 \
  --atrous_rates=36 \
  --output_stride=8 \
  --vis_crop_size="537,561" \
  --decoder_output_stride=4 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${ROSETTE_DATASET}" \
  --add_flipped_images=True \
  --also_save_raw_predictions=true \
  --max_number_of_iterations=1

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${WORK_DIR}"/export_model.py \
  --dataset="ara_rosettes" \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=3 \
  --crop_size=537 \
  --crop_size=561 \
  --add_flipped_images=True
#  --inference_scales="[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]"
#  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
