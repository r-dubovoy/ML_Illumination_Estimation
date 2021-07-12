import os
from tensorflow import keras
import numpy as np
from utils.terminal_text import bcolors as _bcolors
import cv2
from dp_model01 import model_Parallel_regression
from dp_model01 import data_manip


if not 'ML_LIGHT_DIR' in os.environ:
    os.system("say %s" % ("Create root environment variable"))
    raise Exception("Create root environment variable")

root = os.environ['ML_LIGHT_DIR']

SCENE = root + 'training/blender/scenes/ruslan-with-hat.blend'
SCRIPT = root + 'training/python/scripts/render_single_random.py'

def test_one_by_one(model):
    print("Provide path to make a prediction. Input q to quit.")
    while True:
        image_path = input("Path: ")
        if(image_path == 'q' or image_path == 'quit'):
            break
        else:
            if(os.path.exists(image_path)):
                print(image_path)
                colors, label, _ = data_manip.prep_single(image_path)
                model.predict_single_compare(colors, label)
            else:
                print(_bcolors.FAIL +
                      "File doesn't exist. Provide correct path."
                      + _bcolors.ENDC)


def get_newest(dir):
    os.chdir(dir)
    files = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
    return files[-1]

"""Render-Predict-Render"""
def rpr_random(model):
    # Render and show test subject
    file_name = 'before'
    render_path = root + 'training/blender/render_single/layers_upd/'
    SCENE = root + 'training/blender/scenes/ruslan-with-hat-layers.blend'
    SCRIPT = root + 'training/python/scripts/layers_before_random.py'
    exec = os.system(
        'blender --background {} '.format(SCENE) +
        '--python {} -- {} '.format(SCRIPT, file_name))

    newest = get_newest(render_path)
    img = cv2.imread(newest)
    cv2.imshow('before', img)
    cv2.waitKey(0)

    colors, label, _ = data_manip.prep_single(newest)
    prediction = model.predict_single(colors)

    # Render and show prediction

    file_name = 'after'
    SCENE = root + 'training/blender/scenes/ruslan-with-hat-layers.blend'
    SCRIPT = root + 'training/python/scripts/layers_after.py'
    exec = os.system(
        'blender --background {} '.format(SCENE) +
        '--python {} -- {} {} {} {} {} {} {} '
        .format(SCRIPT, file_name, label[0], label[1], label[2], prediction[0],
                                                      prediction[1], prediction[2]))
    newest = get_newest(render_path)
    img = cv2.imread(newest)
    cv2.imshow('after', img)
    cv2.waitKey(0)


# """Render-Predict-Render"""
# def rpr_random(model):
#
#     # Render and show test subject
#     file_name = 'before'
#     render_path = root + 'training/blender/render_single/point 80 W (5m)/{}'.format(file_name)
#     SCENE = root + 'training/blender/scenes/ruslan-with-hat.blend'
#     SCRIPT = root + 'training/python/scripts/render_single_random.py'
#     exec = os.system(
#     'blender --background {} '.format(SCENE) +
#     '--python {} -- {} '.format(SCRIPT, file_name))
#     img = cv2.imread(render_path + '.jpg')
#     cv2.imshow('before', img)
#     cv2.waitKey(0)
#
#     colors, label, _ = data_manip.prep_single(render_path + '.jpg')
#     prediction = model.predict_single(colors)
#
#     # Render and show prediction
#
#     file_name = 'after'
#     render_path = root + 'training/blender/render_single/point 80 W (5m)/{}'.format(file_name)
#     SCENE = root + 'training/blender/scenes/ruslan-with-hat.blend'
#     SCRIPT = root + 'training/python/scripts/render_single_argv.py'
#     print(_bcolors.FAIL + render_path + '.jpg' + _bcolors.ENDC)
#     exec = os.system(
#     'blender --background {} '.format(SCENE) +
#     '--python {} -- {} {} {} {} '.format(SCRIPT, prediction[0], prediction[1], prediction[2], file_name))
#     img = cv2.imread(render_path + '.jpg')
#     cv2.imshow('before', img)
#     cv2.waitKey(0)