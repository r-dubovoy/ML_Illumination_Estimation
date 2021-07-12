import sys
import os
import tensorflow as tf
from tensorflow import keras
PROJ_DIR = os.environ['ML_LIGHT_DIR']
model01_path = os.path.abspath(os.path.join(PROJ_DIR, 'training/python/model01/'))
sys.path.append(model01_path)
print(PROJ_DIR)
print(model01_path)
import model_Parallel_regression

MODEL_PATH = model01_path + ''

model = model_Parallel_regression.Parallel(load_path='')