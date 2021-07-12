import numpy as np
import pickle
from dp_model01 import landmark_utils
import cv2
import os
import dlib
from utils.terminal_text import bcolors as _bcolors
from matplotlib import pyplot as plt
from scipy import interpolate

# detector to detect faces
_detector = dlib.get_frontal_face_detector()
# predictor to predict landmarks
_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_label(fname):
    fname_split = fname.split()
    if len(fname_split) < 3:
        return np.array(["no_label"])
    x_var = fname_split[0]
    y_var = fname_split[1]
    z_var = fname_split[2]
    # print(" x: {}\n y: {}\n z: {}\n".format(x_var, y_var, z_var))
    return np.array([x_var, y_var, z_var])


"""
parse single image
"""
def import_raw_single(fpath):
    fname = fpath.split('/')[-1]
    label = get_label(fname)
    img = cv2.imread(fpath)
    landmarks, _ = landmark_utils.place_landmarks_single(detector=_detector, predictor=_predictor, img=img)
    lightpoints = landmark_utils.calc_additional_points(landmarks)
    colors_gray = landmark_utils.extract_colors_grayscale(lightpoints, img, 3)
    # colors = landmark_utils.extract_colors(lightpoints, img, 3)
    return colors_gray, label, lightpoints

"""
import raw data from the folder
"""
def import_raw_data(data_dir):
    data_colors = np.empty((0, landmark_utils.LIGHTPOINT_NUMBER))
    data_labels = np.empty((0, 3))
    data_lightpoints = np.empty((0, landmark_utils.LIGHTPOINT_NUMBER, 2))
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            label = get_label(filename)
            img = cv2.imread(data_dir+filename)
            if(img.any()):
                print(_bcolors.OKBLUE + "reading {}".format(data_dir + filename) + _bcolors.ENDC)
            else:
                print(_bcolors.FAIL + "! Couldn't read {}".format(data_dir + filename) + _bcolors.ENDC)
            landmarks, faces_found = landmark_utils.place_landmarks_single(detector=_detector, predictor=_predictor, img=img)
            if(faces_found):
                # print(_bcolors.WARNING + "extracted landmarks" + _bcolors.ENDC)
                lightpoints = landmark_utils.calc_additional_points(landmarks)
                # print(_bcolors.WARNING + "calculated lightpoints" + _bcolors.ENDC)
                # colors = landmark_utils.extract_colors(lightpoints, img, 3)
                colors_gray = landmark_utils.extract_colors_grayscale(lightpoints, img, 3)
                # print(_bcolors.WARNING + "extracted colors" + _bcolors.ENDC)
                data_colors = np.append(data_colors, np.array([colors_gray]), axis = 0)
                # print(_bcolors.WARNING + "appended colors" + _bcolors.ENDC)
                data_labels = np.append(data_labels, np.array([label]), axis=0)
                # print(_bcolors.WARNING + "appended labels" + _bcolors.ENDC)
                data_lightpoints = np.append(data_lightpoints, np.array([lightpoints]), axis = 0)
                # print(_bcolors.WARNING + "appended lightpoints" + _bcolors.ENDC)
    return data_colors, data_labels, data_lightpoints


"""
dump data
(arr1, arr2, arr3, filename1, filename2, filename3)
"""
def pickle_data(colors, labels, lightpoints, colors_filename, labels_filename, lightpoints_filename):
    with open(colors_filename, 'wb') as f:
        pickle.dump(colors, f)
    with open(labels_filename, 'wb') as f:
        pickle.dump(labels, f)
    with open(lightpoints_filename, 'wb') as f:
        pickle.dump(lightpoints, f)

def save_np_data(colors, labels, lightpoints, colors_filename, labels_filename, lightpoints_filename):
    np.save(colors_filename, colors)
    np.save(labels_filename, labels)
    if lightpoints_filename != "":
        np.save(lightpoints_filename, lightpoints)
    print(_bcolors.OKGREEN + "saved {} entities".format(colors.shape[0]) + _bcolors.ENDC)

"""
load data
"""
def unpickle_data(colors_filename, labels_filename, lightpoints_filename):
    with open(colors_filename, 'rb') as f:
        colors = pickle.load(f)
    with open(labels_filename, 'rb') as f:
        labels = pickle.load(f)
    with open(lightpoints_filename) as f:
        lightpoints = pickle.load(f)
    return colors, labels, lightpoints

def load_np_data(colors_filename, labels_filename, lightpoints_filename):
    colors = np.load(colors_filename)
    labels = np.load(labels_filename)
    lightpoints = np.load(lightpoints_filename)
    print(_bcolors.OKGREEN + "loaded {}"
                     " entities".format(colors.shape[0]) + _bcolors.ENDC)
    return colors, labels, lightpoints

"""
Reverse channel order, ex: (256,256,3) -> (3, 256, 256)
"""
def reverse_channel_order(arr):
    shape = arr.shape
    arr = np.swapaxes(arr, len(shape)-2, len(shape)-1)
    return arr

"""Convert one image in ready to use data"""
def prep_single(image_path):
    colors, label, lightpoints = import_raw_single(image_path)
    # print(_bcolors.FAIL + "label: {}".format(label) + _bcolors.ENDC)
    colors = reverse_channel_order(colors)
    colors = np.expand_dims(colors, axis=0)
    return colors, label, lightpoints



"""Model data"""
def plot_train_process(training_data):
    plot_smooth(training_data.history['loss'], label='MAE (training data)')
    plot_smooth(training_data.history['val_loss'], label='MAE (validation data)')
    plt.legend(loc="upper left")
    plt.ylabel('MAE value')
    plt.xlabel('No. epoch')
    plt.ylim(0, 10)
    plt.show()

def plot_smooth(y, label):
    x_new = np.linspace(0, len(y), 300)
    a_BSpline = interpolate.make_interp_spline(list(range(0, len(y))), y)
    y_new = a_BSpline(x_new)
    plt.plot(x_new, y_new, label=label)