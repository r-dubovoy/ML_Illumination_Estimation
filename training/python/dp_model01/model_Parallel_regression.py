from tensorflow import keras
from utils.terminal_text import bcolors as _bcolors
import numpy as np
from dp_model01.data_manip import plot_train_process
import time


class Parallel:
    def __init__(self, input_shape=0, hidden_units=0, load_path = '',
                 learning_rate = 0.01):
        print(_bcolors.BOLD + "input shape of a model is: "
                              "{}".format(input_shape) + _bcolors.ENDC)
        if load_path != '':
            self.model = keras.models.load_model(load_path)
            print(_bcolors.OKGREEN + 'loaded model: {}'.format(load_path)
                  + _bcolors.ENDC)
        else:
            input_layer = keras.layers.Input(shape=(input_shape))

            hidden_layer_x = keras.layers.Dense(hidden_units,
                                                activation = 'relu')(input_layer)
            out_layer_x = keras.layers.Dense(1)(hidden_layer_x)

            hidden_layer_y = keras.layers.Dense(hidden_units,
                                                activation='relu')(input_layer)
            out_layer_y = keras.layers.Dense(1)(hidden_layer_y)

            hidden_layer_z = keras.layers.Dense(hidden_units,
                                                activation='relu')(input_layer)
            out_layer_z = keras.layers.Dense(1)(hidden_layer_z)

            merge_layer = keras.layers.concatenate([out_layer_x,
                                                    out_layer_y, out_layer_z])
            out_layer_final = keras.layers.Dense(3)(merge_layer)

            self.model = keras.Model(inputs=[input_layer],
                                     outputs=[out_layer_final])
            self.model.compile(loss='mae', optimizer='adam')

    def info(self):
        self.model.summary()

    def train(self, X_train, X_test, y_train, y_test, epochs):
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=8, validation_data=(X_test, y_test))
        plot_train_process(history)

    def eval(self, test_data, test_labels):
        self.model.evaluate(test_data, test_labels)

    def save(self, filename):
        self.model.save(filename)
        print(_bcolors.OKGREEN + "model saved as {}".format(filename) + _bcolors.ENDC)

    def predict_single_compare(self, data, label):
        predictions = self.model.predict(data)
        prediction = predictions[0]
        prediction = np.round(prediction, 3)
        label = label.astype(float)
        # label = np.round(label, 3)
        print(_bcolors.BOLD + "%12s" % "RES: " + _bcolors.ENDC + "X: {:.3f} Y: {:.3f} Z: {:.3f}".format(prediction[0], prediction[1], prediction[2]))
        print(_bcolors.BOLD + "%12s" %  "EXPECTED: " + _bcolors.ENDC + "X: {:.3f} Y: {:.3f} Z: {:.3f}".format(label[0], label[1], label[2]))
        print(_bcolors.BOLD + "%12s" % "DIFF: " + _bcolors.ENDC + "X: {:.3f} Y: {:.3f} Z: {:.3f}".format(label[0]-prediction[0], label[1]-prediction[1], label[2]-prediction[2]))

    def predict_single(self, data):
        start = time.time()
        prediction = self.model.predict(data)[0]
        prediction = np.round(prediction, 5)
        end = time.time()
        print("Prediction time elapsed: " + _bcolors.BOLD + str(end-start) + _bcolors.ENDC)
        return prediction

