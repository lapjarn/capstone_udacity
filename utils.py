# -*- coding: utf-8 -*-
# utils.py

""" 
  This module contains helper functions to load the Cifar100 dataset and
  to load the corresponding label names.
"""

import os
import tarfile
import urllib.request
import pickle
import numpy as np

_url = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
_dir_name = './data'
_file_name = 'cifar-100-python.tar.gz'
_folder_name = 'cifar-100-python'

def _unpickle(file):
  fo = open(file, 'rb')
  _dict = pickle.load(fo, encoding='bytes')
  fo.close()
  return _dict
  
def _load_data(file):
  data = []
  labels = []
  
  _dict = _unpickle(file)
  
  data = _dict[b'data']
  coarse_labels = _dict[b'coarse_labels']
  fine_labels = _dict[b'fine_labels']
  num_data_points = len(data)
  
  data = data.reshape(num_data_points, 3, 32, 32)
  data = np.transpose(data, [0, 2, 3, 1])
  
  return data, np.array(coarse_labels), np.array(fine_labels)
  
def load_dataset():
  """ Loads Cifar100 dataset by downloading and extracting if required.

  Args:
    None

  Returns:
    train_data, train_coarse_label, train_fine_label, test_data, test_coarse_label, test_fine_label
  """
  file_path = os.path.join(_dir_name, _file_name)
  folder_path = os.path.join(_dir_name, _folder_name)
  
  if not os.path.exists(file_path):
    os.makedirs(_dir_name)
    print("Downloading CIFAR100 dataset...")
    try:
      urllib.request.urlretrieve(_url, file_path)
    except:
      print("Failed to download CIFAR100 dataset.")      
    print("Download complete.")
  else:
    print("Dataset already downloaded. Did not download twice.")
    
  if not os.path.exists(os.path.abspath(folder_path)):
    print("Extracting files...")
    tarfile.open(file_path, 'r:gz').extractall(_dir_name)
    print("Extraction successfully done to {}.".format(folder_path))
  else:
    print("Dataset already extracted. Did not extract twice.")
    
  train_file = os.path.join(folder_path, 'train')
  test_file = os.path.join(folder_path, 'test')
  
  X, Y_c, Y_f = _load_data(train_file)
  Xt, Yt_c, Yt_f = _load_data(test_file)
  
  return X, Y_c, Y_f, Xt, Yt_c, Yt_f
  
def load_label_names():
  """ Loads Cifar100 meta data and return label names.

  Args:
    None

  Returns:
    coarse_label_names, fine_label_names
  """
  folder_path = os.path.join(_dir_name, _folder_name)
  
  meta_file = os.path.join(folder_path, 'meta')
  _dict = _unpickle(meta_file)
  
  coarse_label_names = _dict[b'coarse_label_names']
  fine_label_names = _dict[b'fine_label_names']
  
  return np.array(coarse_label_names), np.array(fine_label_names)
