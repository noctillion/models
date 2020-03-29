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
# - this is a modified version of download_and_convert_voc2012.sh
#
# Script to download and preprocess plant rosette datasets.
#
# Usage:
#   bash ./download_and_convert_ara_senescentSet.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_rosette_data.py
#     - download_and_convert_ara_senescentSet.sh
#     + ara_senescentSet
#       + rosettes
#         + JPEGImages
#         + SegmentationClass
#

# Exit immediately if a command exits with a non-zero status.
set -e

DATASET="ara_senescentSet"

CURRENT_DIR=$(pwd)
WORK_DIR="./${DATASET}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Helper function to download and unpack ara_senescentSet dataset.
download_and_uncompress() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    wget -O "${DATASET}.tar.gz" -nd -c "${BASE_URL}/${FILENAME}"
  fi
  echo "Uncompressing ${FILENAME}"
  tar -xzf "${DATASET}.tar.gz"
}

# Download the images.
BASE_URL="https://www.dropbox.com/s/"
FILENAME="7u16iw5i4q2w8j6/ara_senescentSet.tar.gz?dl=1"

download_and_uncompress "${BASE_URL}" "${FILENAME}"

cd "${CURRENT_DIR}"

# Root path for ara_senescentSet dataset.
ROSETTE_ROOT="${WORK_DIR}/rosettes"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${ROSETTE_ROOT}/PNGImages"
LIST_FOLDER="${ROSETTE_ROOT}/ImageSets/Segmentation"
SEMANTIC_SEG_FOLDER="${ROSETTE_ROOT}/SegmentationClassRaw"

echo "Converting ara_senescentSet ..."
python ./build_rosette_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="png" \
  --output_dir="${OUTPUT_DIR}" \
  --shard_number=9
