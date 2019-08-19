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
"""Saves an annotation as one png image.

This script saves an annotation as one png image, and has the option to add
colormap to the png image for better visualization.
"""

import csv
import numpy as np
from PIL import Image,ImageStat
from skimage import measure
import tensorflow as tf

from deeplab.utils import get_dataset_colormap


def count_pixels(label,
                    image,
                    save_dir,
                    filename,
                    add_colormap=True,
		                get_regionprops=False,
		                channelstats=False,
		                save_prediction=False,
                    normalize_to_unit_values=False,
                    scale_values=False,
                    colormap_type=get_dataset_colormap.get_pascal_name()):
  """Counts pixels of all classes and optionally saves the given label to image on disk.

  Args:
    label: The numpy array to be saved. The data will be converted
      to uint8 and saved as png image.
    image: Array representing the original image.
    save_dir: String, the directory to which the results will be saved.
    filename: String, the image filename.
    add_colormap: Boolean, add color map to the label or not.
    save_prediction: Boolean, save the resulting prediction to disk.
    get_regionprops: Boolean, calculate various region properties (see: https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops).
    channelstats: Boolean, calculate channel means in original image within the label mask.
    normalize_to_unit_values: Boolean, normalize the input values to [0, 1].
    scale_values: Boolean, scale the input values to [0, 255] for visualization.
    colormap_type: String, colormap type for visualization.
  """
  # Add colormap for visualizing the prediction.
  if add_colormap:
    colored_label = get_dataset_colormap.label_to_color_image(
        label, colormap_type)
  else:
    colored_label = label
    if normalize_to_unit_values:
      min_value = np.amin(colored_label)
      max_value = np.amax(colored_label)
      range_value = max_value - min_value
      if range_value != 0:
        colored_label = (colored_label - min_value) / range_value

    if scale_values:
      colored_label = 255. * colored_label

  frame = {'file' : filename + '.png', 'total_pixels' : label.size}

  # TODO: don't hardcode label names
  LABEL_NAMES = np.asarray(['background', 'rosette'])
  # iterate over label names and count pixels for each class 
  for idx,labelclass in enumerate(LABEL_NAMES):
    count = np.count_nonzero(label == idx)
    frame[labelclass] = count

  if channelstats:
    org = Image.fromarray(image.astype(dtype=np.uint8))
    mask = Image.fromarray(label.astype(dtype=np.uint8))
    stat = ImageStat.Stat(org, mask=mask)

    bands = np.asarray(['red', 'green','blue'])
    for idx,band in enumerate(bands):
      frame[band + "_sum_within_mask"] = stat.sum[idx]
      if stat.count[idx] != 0:
        frame[band + "_mean_within_mask"] = stat.mean[idx]
      else:
        frame[band + "_mean_within_mask"] = 'NA'
      frame[band + "_median_within_mask"] = stat.median[idx]
    
  if get_regionprops:
    traits = np.asarray(['filled_area','convex_area',"equivalent_diameter","major_axis_length","minor_axis_length","perimeter","eccentricity","extent","solidity"])
    properties = measure.regionprops(label)
    for trait in traits:
      if len(properties) == 1:
        frame[trait] = properties[0][trait]
      else:
        frame[trait] = 'NA' 

  if save_prediction:
    pil_image = Image.fromarray(colored_label.astype(dtype=np.uint8))
    with tf.gfile.Open('%s/%s.png' % (save_dir, filename), mode='w') as f:
      pil_image.save(f, 'PNG')


    # write pixel counts to tsv file
  with open(save_dir + '/pixel_counts.tsv', 'a') as counts:
    Writer = csv.DictWriter(counts, delimiter=' ', fieldnames=frame.keys())
    if not counts.tell():
      Writer.writeheader()
    Writer.writerow(frame)

  return frame
