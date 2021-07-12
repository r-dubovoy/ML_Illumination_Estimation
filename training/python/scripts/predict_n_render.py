import sys
import os
import tensorflow as tf
import bpy
import cv2
import random
from tensorflow import keras
from dp_model01 import model_Parallel_regression
from dp_model01 import data_manip

IMG_BEFORE = 'to_predict'
IMG_AFTER = 'predicted'

"""Render random image to predict"""
x = random.uniform(-3, 3)
y = random.uniform(-3, -0.5)
z = random.uniform(0.5, 3)
light = bpy.context.scene.objects.get('Light01')
light.location.x = x
light.location.y = y
light.location.z = z
model = model_Parallel_regression('../dp_model01/model_Feb23')
bpy.context.scene.render.filepath = IMG_BEFORE
bpy.ops.render.render(write_still = True)
img = cv2.imread(IMG_BEFORE + 'jpg')
cv2.imshow('before', img)
cv2.waitKey(0)

"""Make x y z for hat prediction"""
colors, label, lightpoints = data_manip.import_raw_single(IMG_BEFORE + '.jpg')
prediction = model.predict_single_compare(lightpoints, [0, 0, 0])

"""Render prediction"""
light = bpy.context.scene.objects.get('Light02')
light.location.x = prediction[0]
light.location.y = prediction[1]
light.location.z = prediction[2]

bpy.context.scene.render.filepath = IMG_BEFORE
bpy.ops.render.render(write_still = True)

"""Show prediciton"""
img = cv2.imread(IMG_AFTER + '.jpg')
cv2.imshow('after', img)
cv2.waitKey(0)