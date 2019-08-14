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
#python "${WORK_DIR}"/model_test.py -v

# Go to datasets folder and download ROSETTE segmentation dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
ROSETTE_FOLDER="ara_rosetteSet"
EXP_FOLDER="exp/train_on_xception65_imagenet_multi_gradient"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/${EXP_FOLDER}/train"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ROSETTE_FOLDER}/${EXP_FOLDER}/vis"
#mkdir -p "${TRAIN_LOGDIR}"
#mkdir -p "${VIS_LOGDIR}"


ROSETTE_DATASET="${WORK_DIR}/${DATASET_DIR}/testdat/shards"
#ROSETTE_DATASET="${WORK_DIR}/${DATASET_DIR}/results"

# Visualize the results.
python "${WORK_DIR}"/predict_rosettes.py \
  --dataset="ara_rosettes" \
  --logtostderr \
  --vis_split="test" \
  --model_variant="xception_65" \
  --atrous_rates=12 \
  --atrous_rates=24 \
  --atrous_rates=36 \
  --output_stride=8 \
  --vis_crop_size="537,561" \
  --decoder_output_stride=4 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${ROSETTE_DATASET}" \
  --dataset_dir="${ROSETTE_DATASET}" \
  --add_flipped_images=True \
  --max_number_of_iterations=1
