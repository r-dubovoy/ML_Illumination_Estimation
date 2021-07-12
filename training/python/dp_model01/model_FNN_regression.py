from tensorflow import keras
from utils.terminal_text import bcolors as _bcolors
import numpy as np
from dp_model01.data_manip import plot_train_process
from matplotlib import pyplot as plt


class FNN:
    def __init__(self, input_shape=0, layer_units=[0], output_layer=0,
                 load_path = '', learning_rate = 0.01, optimizer = 'adam',
                 loss = 'mean_square_error'):
        print(_bcolors.BOLD + "input shape of a model is:"
                              " {}".format(input_shape) + _bcolors.ENDC)
        if load_path != '':
            self.model = keras.models.load_model(load_path)
            print(_bcolors.OKGREEN + 'loaded model: '
                                     '{}'.format(load_path) + _bcolors.ENDC)
        else:
            self.model = keras.models.Sequential()
            self.model.add(keras.layers.Input(shape=(input_shape)))
            for unit in layer_units:
                # if(len(self.model.layers) == 0):
                #     self.model.add(keras.layers.Dense(unit, activation = 'relu', input_shape = input_shape))
                # else:
                self.model.add(keras.layers.Dense(unit, activation = 'relu'))
            #if using 3 channels
            # self.model.add(keras.layers.Dense(output_layer, activation = 'relu'))
            # self.model.add(keras.layers.Flatten())
            # self.model.add(keras.layers.Dense(output_layer, activation = 'linear'))
            self.model.add(keras.layers.Dense(output_layer))
            self.model.compile(optimizer = optimizer,
                               loss = loss,
                               metrics = ['accuracy'], learning_rate=learning_rate)


            # self.model.compile(loss='mae', optimizer='adam')

    def info(self):
        self.model.summary()

    def train(self, X_train, X_test, y_train, y_test, epochs):
        history = self.model.fit(X_train, y_train, epochs=epochs,
                                 batch_size=8, validation_data=(X_test, y_test))
        plot_train_process(history)

    def eval(self, test_data, test_labels):
        self.model.evaluate(test_data, test_labels)

    def save(self, filename):
        self.model.save(filename)

    def predict_single_compare(self, data, label):
        predictions = self.model.predict(data)
        prediction = predictions[0]
        prediction = np.round(prediction, 3)
        label = label.astype(float)
        # label = np.round(label, 3)
        print(_bcolors.BOLD + "%12s" % "RES: " + _bcolors.ENDC +
              "X: {:.3f} Y: {:.3f} Z: {:.3f}"
                "".format(prediction[0], prediction[1], prediction[2]))
        print(_bcolors.BOLD + "%12s" %  "EXPECTED: " + _bcolors.ENDC +
              "X: {:.3f} Y: {:.3f} Z: {:.3f}"
                "".format(label[0], label[1], label[2]))
        print(_bcolors.BOLD + "%12s" % "DIFF: " + _bcolors.ENDC +
              "X: {:.3f} Y: {:.3f} Z: {:.3f}"
              "".format(label[0]-prediction[0],
                        label[1]-prediction[1], label[2]-prediction[2]))



