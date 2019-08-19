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

DATASET_DIR="datasets"

# Set up the working directories.
ROSETTE_FOLDER="ara_rosetteSet"
EXP_FOLDER="exp/train_on_xception65_imagenet"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/${EXP_FOLDER}/eval"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"

ROSETTE_DATASET="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/tfrecord"

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
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --eval_scales=1.0 \
  --dataset_dir="${ROSETTE_DATASET}" \
  --max_number_of_evaluations=0

#  --add_flipped_images=True \
#  --eval_scales=1.0 \
#  --labels_offset=1 \
