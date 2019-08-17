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

"""Segmentation results visualization on a given set of images.

See model.py for more details and usage.
"""
import csv
import os.path
import time
import numpy as np
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import data_generator
from deeplab.utils import count_pixels

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for visualizing the model.

flags.DEFINE_integer('vis_batch_size', 1,
                     'The number of images in each batch during visualization.')

flags.DEFINE_list('vis_crop_size', '513,513',
                  'Crop size [height, width] for visualization.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

flags.DEFINE_integer(
    'quantize_delay_step', -1,
    'Steps to start quantized training. If < 0, will not quantize model.')

# Dataset settings.

flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('vis_split', 'val',
                    'Which split of the dataset used for visualizing results')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_enum('colormap_type', 'pascal', ['pascal', 'cityscapes'],
                  'Visualization colormap type.')

flags.DEFINE_integer('max_number_of_iterations', 0,
                     'Maximum number of visualization iterations. Will loop '
                     'indefinitely upon nonpositive values.')

# The folder where semantic segmentation predictions are saved.
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'

# The format to save image.
_IMAGE_FORMAT = '%06d_image'

# The format to save prediction
_PREDICTION_FORMAT = '%06d_prediction'

def _process_batch(sess, original_images, semantic_predictions, image_names,
                   image_heights, image_widths, image_id_offset, save_dir):
  """Evaluates one single batch qualitatively.

  Args:
    sess: TensorFlow session.
    original_images: One batch of original images.
    semantic_predictions: One batch of semantic segmentation predictions.
    image_names: Image names.
    image_heights: Image heights.
    image_widths: Image widths.
    image_id_offset: Image id offset for indexing images.
    save_dir: The directory where the predictions will be saved.
  """
  (original_images,
   semantic_predictions,
   image_names,
   image_heights,
   image_widths) = sess.run([original_images, semantic_predictions,
                             image_names, image_heights, image_widths])

  num_image = semantic_predictions.shape[0]
  for i in range(num_image):
    image_height = np.squeeze(image_heights[i])
    image_width = np.squeeze(image_widths[i])
    original_image = np.squeeze(original_images[i])
    semantic_prediction = np.squeeze(semantic_predictions[i])
    crop_semantic_prediction = semantic_prediction[:image_height, :image_width]

    # Save image.
#    save_annotation.save_annotation(
#        original_image, save_dir, image_names[i],
#        add_colormap=False)

    # Save prediction.
#    save_annotation.save_annotation(
#        crop_semantic_prediction, save_dir,
#        "%s" % image_names[i].decode('utf-8'), add_colormap=True,
#        colormap_type=FLAGS.colormap_type)

    line = count_pixels.count_pixels(
        crop_semantic_prediction, original_image, save_dir,
        "%s" % image_names[i].decode('utf-8'), add_colormap=True, save_prediction=True,
	      regionprops=True, channelstats=True, colormap_type=FLAGS.colormap_type)

    # write pixel counts to tsv file   
    with open(save_dir + '/pixel_counts.csv', 'a') as counts:
      Writer = csv.DictWriter(counts, delimiter=' ', fieldnames=line.keys())
      if not counts.tell():
        Writer.writeheader() 
        Writer.writerow(line)

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Get dataset-dependent information.
  dataset = data_generator.Dataset(
      dataset_name=FLAGS.dataset,
      split_name=FLAGS.vis_split,
      dataset_dir=FLAGS.dataset_dir,
      batch_size=FLAGS.vis_batch_size,
      crop_size=[int(sz) for sz in FLAGS.vis_crop_size],
      min_resize_value=FLAGS.min_resize_value,
      max_resize_value=FLAGS.max_resize_value,
      resize_factor=FLAGS.resize_factor,
      model_variant=FLAGS.model_variant,
      is_training=False,
      should_shuffle=False,
      should_repeat=False)

  # Prepare for visualization.
  tf.gfile.MakeDirs(FLAGS.vis_logdir)
  save_dir = os.path.join(FLAGS.vis_logdir, _SEMANTIC_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(save_dir)

  tf.logging.info('Visualizing on %s set', FLAGS.vis_split)

  with tf.Graph().as_default():
    samples = dataset.get_one_shot_iterator().get_next()

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
        crop_size=[int(sz) for sz in FLAGS.vis_crop_size],
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions = model.predict_labels(
          samples[common.IMAGE],
          model_options=model_options,
          image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      if FLAGS.quantize_delay_step >= 0:
        raise ValueError(
            'Quantize mode is not supported with multi-scale test.')
      predictions = model.predict_labels_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)
    predictions = predictions[common.OUTPUT_TYPE]

    if FLAGS.min_resize_value and FLAGS.max_resize_value:
      # Only support batch_size = 1, since we assume the dimensions of original
      # image after tf.squeeze is [height, width, 3].
      assert FLAGS.vis_batch_size == 1

      # Reverse the resizing and padding operations performed in preprocessing.
      # First, we slice the valid regions (i.e., remove padded region) and then
      # we resize the predictions back.
      original_image = tf.squeeze(samples[common.ORIGINAL_IMAGE])
      original_image_shape = tf.shape(original_image)
      predictions = tf.slice(
          predictions,
          [0, 0, 0],
          [1, original_image_shape[0], original_image_shape[1]])
      resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]),
                                   tf.squeeze(samples[common.WIDTH])])
      predictions = tf.squeeze(
          tf.image.resize_images(tf.expand_dims(predictions, 3),
                                 resized_shape,
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                 align_corners=True), 3)

    tf.train.get_or_create_global_step()
    if FLAGS.quantize_delay_step >= 0:
      tf.contrib.quantize.create_eval_graph()

    num_iteration = 0
    max_num_iteration = FLAGS.max_number_of_iterations

    checkpoints_iterator = tf.contrib.training.checkpoints_iterator(
        FLAGS.checkpoint_dir, min_interval_secs=FLAGS.eval_interval_secs)
    for checkpoint_path in checkpoints_iterator:
      num_iteration += 1
      tf.logging.info(
          'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                       time.gmtime()))
      tf.logging.info('Visualizing with model %s', checkpoint_path)

      scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())
      session_creator = tf.train.ChiefSessionCreator(
          scaffold=scaffold,
          master=FLAGS.master,
          checkpoint_filename_with_path=checkpoint_path)
      with tf.train.MonitoredSession(
          session_creator=session_creator, hooks=None) as sess:
        batch = 0
        image_id_offset = 0

        while not sess.should_stop():
          tf.logging.info('Visualizing batch %d', batch + 1)
          _process_batch(sess=sess,
                         original_images=samples[common.ORIGINAL_IMAGE],
                         semantic_predictions=predictions,
                         image_names=samples[common.IMAGE_NAME],
                         image_heights=samples[common.HEIGHT],
                         image_widths=samples[common.WIDTH],
                         image_id_offset=image_id_offset,
                         save_dir=save_dir)
          image_id_offset += FLAGS.vis_batch_size
          batch += 1

      tf.logging.info(
          'Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                       time.gmtime()))
      if max_num_iteration > 0 and num_iteration >= max_num_iteration:
        break

if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('vis_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
