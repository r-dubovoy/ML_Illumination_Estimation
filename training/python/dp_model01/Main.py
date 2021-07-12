import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from utils.terminal_text import bcolors as _bcolors
from dp_model01 import data_manip
from dp_model01 import model_FNN_regression
from dp_model01 import model_Parallel_regression
from dp_model01 import interfaces
from dp_model01 import landmark_utils
from sklearn.model_selection import train_test_split
import cv2



os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
Test single.
"""
# image_path = "/Users/romandubovyi/ml_jedis/ml_light/training/blender/render/test_data/-0.859641229395554 -2.509287688501131 1.008455096007388 1601468512.591806.jpg"
# image_path = 'ruslan03.jpg'
# img = cv2.imread(image_path)
# colors, label, lightpoints = data_manip.import_raw_single(image_path)
# landmark_utils.show_Gray(img, colors, lightpoints, image_path)

"""
Process data and save.
"""
# batch_name = "uniform320_GRAY_Feb18"
# data_path = "/Users/romandubovyi/ml_jedis/ml_light/training/blender/render/ruslan 300 W uniform 320x320/"
# colors, labels, lightpoints = data_manip.import_raw_data(data_path)
# data_manip.save_np_data(colors, labels, lightpoints, "{}_data".format(batch_name), "{}_labels".format(batch_name), "{}_lightpoints".format(batch_name))

"""
Load data.
"""
batch_name = "uniform320_GRAY_Feb18"
colors, labels, lightpoints = data_manip.load_np_data("{}_data.npy".format(batch_name),
                                                      "{}_labels.npy".format(batch_name),
                                                      "{}_lightpoints.npy".format(batch_name))

print(colors.shape)
print(labels.shape)
print(lightpoints.shape)

"""
Train model and save model.
"""
# colors = data_manip.reverse_channel_order(colors)
labels = labels.astype(float)
# print(_bcolors.BOLD + "input.shape: {}".format(colors[0].shape) + _bcolors.ENDC)
model = model_Parallel_regression.Parallel(input_shape=colors.shape[1], hidden_units=79)
model.info()


X_train, X_test, y_train, y_test = train_test_split(colors, labels,
                                                    test_size=0.3, shuffle=True)

# model = model_FNN_regression.FNN(input_shape=colors.shape[1],
#                                  layer_units=np.array([64, 32]),
#                                  output_layer=3, learning_rate=0.001)

# model.train(X_train, X_test, y_train, y_test, 20)
# model.save("model_name")
# interfaces.test_one_by_one(model)

"""Load model"""

# model = model_Parallel_regression.Parallel(load_path='my_model')

"""Test model 'one-by-one' method"""

# interfaces.test_one_by_one(model)

"""Render predict render"""

# model = model_Parallel_regression.Parallel(load_path='../dp_model01/model_Feb25')
#
# # interfaces.test_one_by_one(model)
# interfaces.rpr_random(model)