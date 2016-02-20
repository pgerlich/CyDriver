import gzip
import os
import numpy as np
import cv2
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):

    #Reshape images
    images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
    labels = labels.reshape(labels.shape[0], labels.shape[1])

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = len(images)

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()

  all_images = read_images(train_dir)
  all_labels = read_labels("", all_images)

  TRAIN_SIZE = (len(all_images) * 3 ) / 4
  TEST_SIZE = len(all_images) / 8
  VALIDATION_SIZE = len(all_images) / 8

  train_images = np.asarray(all_images[:TRAIN_SIZE]) #Grab training images
  test_images = np.asarray(all_images[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]) #Grab test images
  validation_images = np.asarray(all_images[TRAIN_SIZE + TEST_SIZE:]) #Grab validation images

  train_labels= np.asarray(all_labels[:TRAIN_SIZE]) #Grab training images
  test_labels = np.asarray(all_labels[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]) #Grab test images
  validation_labels = np.asarray(all_labels[TRAIN_SIZE + TEST_SIZE:]) #Grab validation images

  data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
  data_sets.validation = DataSet(validation_images, validation_labels, dtype=dtype)
  data_sets.test = DataSet(test_images, test_labels, dtype=dtype)
  data_sets._num_examples = len(all_images)

  return data_sets

def read_images(directory):
  images = []

  imgDir = os.listdir(directory)

  for image_name in imgDir:
    image = cv2.imread(os.path.join(directory, image_name), cv2.IMREAD_GRAYSCALE)
    images.append(image)

  return images

def read_labels(directory, images):
  labels = []

  index = 0
  for image in images:
    label = numpy.zeros(3)
    label[index % 3] = (index % 3) + 1
    index = index + 1
    labels.append(label)

  return labels




