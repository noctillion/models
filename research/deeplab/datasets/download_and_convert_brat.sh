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
# Script to download and preprocess the BRAT dataset.
#
# Usage:
#   bash ./download_and_convert_brat.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_brat_data.py
#     - download_and_convert_brat.sh
#     - remove_gt_colormap_brat.py
#     + brat
#       + BRATset
#         + deepBRAT
#           + PNGImages
#           + JPEGImages
#           + SegmentationClass
#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./brat"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Helper function to download and unpack BRAT dataset.
download_and_uncompress() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  fi
  echo "Uncompressing ${FILENAME}"
  tar -xf "${FILENAME}"
}

# Download the images.
BASE_URL="http://homepage.univie.ac.at/huetherp88/"
FILENAME="BRAT_TrainingSet.tar.gz"

download_and_uncompress "${BASE_URL}" "${FILENAME}"

cd "${CURRENT_DIR}"

# Root path for BRAT dataset.
BRAT_ROOT="${WORK_DIR}/BRATset/deepBRAT"

# Remove the colormap in the ground truth annotations.
SEG_FOLDER="${BRAT_ROOT}/SegmentationClass"
SEMANTIC_SEG_FOLDER="${BRAT_ROOT}/SegmentationClassRaw"

echo "Removing the color map in ground truth annotations..."
python ./remove_gt_colormap_brat.py \
  --original_gt_folder="${SEG_FOLDER}" \
  --output_dir="${SEMANTIC_SEG_FOLDER}"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${BRAT_ROOT}/JPEGImages"
LIST_FOLDER="${BRAT_ROOT}/ImageSets/Segmentation"

echo "Converting BRAT dataset..."
python ./build_brat_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"
