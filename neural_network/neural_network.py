from neural_network.cost_functions import SquaredError, CrossEntropy
from neural_network.dense import Dense
from neural_network.digit import Digit
import numpy as np
import pickle
import os


class NeuralNetwork:

    def __init__(self,
                 model=None,
                 cost_function=CrossEntropy):

        # NEURAL NETWORK CORE ATTRIBUTES

        # A list containing Dense objects, that is couples of input/output layers
        # TODO: If no model is given, setting up a single input/output layer, i.e a network without hidden layers.
        #  Make the input layer the right size given a data sample. Initialize the layers in the train method
        self.layers = []

        # If no model is given for the network architecture, initialization takes place in the train method when the
        # network is fed a data sample
        if model is not None:
            self.initialize_layers(model)

        self.cost_function = cost_function

        # TRAINING RELATED ATTRIBUTES

        # Specifying how much time the entirety of the training data set must be fed to the network while training
        self.epochs = 30

        # Specifying the frequency of parameters recalculation (every mini_batch_size pieces of data)
        self.mini_batch_size = 10

        # Specifying how much the parameters much be adjusted each mini_batch_size iterations
        self.learning_rate = 0.5

        # FILE HANDLING ATTRIBUTES

        self.__saving_file = "neural_network_parameters"
        self.__file_extension = "ia"

    def train(self, training_data, evaluation_data,
              cost_function=None, epochs=None, mini_batch_size=None, learning_rate=None):

        len_data = len(training_data)
        len_evaluation_data = len(evaluation_data)

        if cost_function:
            self.cost_function = cost_function

        if epochs:
            self.epochs = epochs

        if mini_batch_size:
            self.mini_batch_size = mini_batch_size

        if learning_rate:
            self.learning_rate = learning_rate

        for i_epoch in range(self.epochs):

            mini_batches = self.generate_mini_batches(training_data, len_data)

            # TODO: erase the loop and feed the process method a vector of mini_batch_size length
            for mini_batch in mini_batches:
                self.process_data_batch(mini_batch)

            print(f"Epoch {i_epoch + 1}: {100 * self.evaluate(evaluation_data) / len_evaluation_data} %")

    def evaluate(self, evaluation_data):
        evaluation_results = [int(np.argmax(self.propagate_forward(x)) == y) for x, y in evaluation_data]
        return sum(evaluation_results)

    def decide(self, image=None, path_to_image=None, array=None):

        array_to_process = None

        if image:
            digit = Digit(image=image)
            array_to_process = digit.array

        elif path_to_image:
            digit = Digit(path_to_image=path_to_image)
            array_to_process = digit.array

        elif array is not None:
            array_to_process = array

        result = self.propagate_forward(array_to_process)

        return np.argmax(result)

    def save(self, filename=None, file_extension=None):

        try:

            count = 0

            if filename:
                self.__saving_file = filename

            if file_extension:
                self.__file_extension = file_extension

            file = self.__saving_file + "." + self.__file_extension

            if os.path.exists(file):
                count = 1

                while os.path.exists(self.__saving_file + f"({count})." + self.__file_extension):
                    count += 1

            answer = input("File " + file + " already exists, do you wish to erase it? (Yes/No)   ").lower()

            if answer == "no":
                file = self.__saving_file + f"({count})." + self.__file_extension

            with open(file=file, mode="wb") as f:
                pickle.dump(obj=self, file=f)

        except:
            print("An error occurred while saving the network parameters.")

    def reinitialize_parameters(self):
        for layer in self.layers:
            layer.reinitialize_parameters()

    def process_data_batch(self, mini_batch):

        for batch in mini_batch:
            y = self.propagate_forward(batch[0])
            self.propagate_backwards(y, batch[1])
            self.update_parameters()

    def propagate_forward(self, x):

        # Formatting data input in a column vector
        x = x.ravel()[:, np.newaxis]

        for layer in self.layers:
            x = layer.propagate_forward(x)

        return x

    def propagate_backwards(self, y, expected_output):

        de_dy = self.cost_function.calculate_derivative(y, expected_output)

        for layer in self.layers[::-1]:
            de_dy = layer.propagate_backwards(de_dy)

    def update_parameters(self):

        for layer in self.layers:
            layer.update_weights(self.learning_rate, self.mini_batch_size)
            layer.update_biases(self.learning_rate, self.mini_batch_size)

    def generate_mini_batches(self, training_data, len_data):
        np.random.shuffle(training_data)
        return [training_data[i * self.mini_batch_size:(i+1) * self.mini_batch_size] for i in range(0, len_data)]

    def initialize_layers(self, model):

        for i in range(len(model) - 1):

            if isinstance(model[i], int):

                if isinstance(model[i + 1], int):
                    self.layers.append(Dense(model[i], model[i + 1]))

                else:
                    self.layers.append(Dense(model[i], model[i + 1][0]))

            else:

                if isinstance(model[i + 1], int):
                    self.layers.append(Dense(model[i][0], model[i + 1], activation_function=model[i][1]))

                else:
                    self.layers.append(Dense(model[i][0], model[i + 1][0], activation_function=model[i][1]))

    # STATIC METHODS
    @staticmethod
    def load_parameters(filename):

        try:

            with open(file=filename, mode="rb") as f:
                nn = pickle.load(file=f)

            return nn

        except:
            print("An error occurred while loading the network parameters.")
