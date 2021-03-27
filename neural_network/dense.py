import numpy as np
from neural_network.activation_functions import Sigmoid


class Dense:

    def __init__(self,
                 number_of_input_nodes=1,
                 number_of_output_nodes=1,
                 activation_function=Sigmoid):

        # Skeleton for the couple of layers
        self.number_of_input_nodes = number_of_input_nodes
        self.number_of_output_nodes = number_of_output_nodes
        self.activation_function = activation_function

        self.x = np.zeros((self.number_of_input_nodes, 1))
        self.v = np.zeros((self.number_of_output_nodes, 1))
        self.y = np.zeros((self.number_of_output_nodes, 1))

        # Setting the initial weights to be in [- initial_weight_scale; + initial_weight_scale]
        self.initial_weight_scale = 1
        self.weights = self.initial_weight_scale * \
                       (np.random.sample((self.number_of_output_nodes, self.number_of_input_nodes)) * 2 - 1)

        # Setting the initial biases to be in [- initial_bias_scale; + initial_bias_scale]
        self.initial_bias_scale = 1
        self.biases = self.initial_bias_scale * \
                      (np.random.sample((self.number_of_output_nodes, 1)) * 2 - 1)

        # Attributes related to parameters adjustments
        self.weights_gradient = np.zeros(self.weights.shape)
        self.biases_gradient = np.zeros(self.biases.shape)

    def propagate_forward(self, x):

        # Storing the input for later back propagation
        self.x = x

        # @ is the matrix multiplication operator
        self.v = self.weights @ self.x + self.biases

        # Computing the activation of the neurons in the output layer
        self.y = self.activation_function.calculate(self.v)

        return self.y

    def propagate_backwards(self, de_dy):

        dy_dv = self.activation_function.calculate_derivative(self.v)
        dv_dw = self.x.transpose()
        dv_dx = self.weights.transpose()
        de_dv = dy_dv * de_dy

        self.weights_gradient += de_dv @ dv_dw
        self.biases_gradient += de_dv

        # The error to propagate backwards
        return dv_dx @ de_dv

    def update_weights(self, learning_rate, mini_batch_size):
        self.weights -= (learning_rate / mini_batch_size) * self.weights_gradient
        self.weights_gradient = np.zeros(self.weights.shape)

    def update_biases(self, learning_rate, mini_batch_size):
        self.biases -= (learning_rate / mini_batch_size) * self.biases_gradient
        self.biases_gradient = np.zeros(self.biases.shape)

    def reinitialize_parameters(self):

        self.x = np.zeros((self.number_of_input_nodes, 1))
        self.v = np.zeros((self.number_of_output_nodes, 1))
        self.y = np.zeros((self.number_of_output_nodes, 1))

        self.weights = self.initial_weight_scale * \
                       (np.random.sample((self.number_of_output_nodes, self.number_of_input_nodes)) * 2 - 1)
        self.biases = self.initial_bias_scale * \
                      (np.random.sample((self.number_of_output_nodes, 1)) * 2 - 1)

        self.weights_gradient = np.zeros(self.weights.shape)
        self.biases_gradient = np.zeros(self.biases.shape)
