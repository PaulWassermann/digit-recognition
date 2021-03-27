from neural_network.neural_network import NeuralNetwork
from neural_network.digit import Digit
import data_loader
import PIL.Image
import numpy as np


def ask_to_retrain(neural_network):

    choice = input("Do you wish to save the parameters? (Yes/No)   ").lower()

    if choice == "yes":
        neural_network.save()

    elif choice == "no":

        second_choice = input("Do you wish to reinitialize the network parameters? (Yes/No)   ").lower()

        training_data, validation_data, test_data = data_loader.load_data_wrapper()

        if second_choice == "yes":
            neural_network.reinitialize_parameters()

        epochs = int(input("Select the number of epochs for the network to train: "))
        neural_network.train(training_data=list(training_data), evaluation_data=list(test_data), epochs=epochs)
        ask_to_retrain(neural_network)


def create_and_train(model=(784, 10), epochs=5):

    training_data, validation_data, test_data = data_loader.load_data_wrapper()

    training_data = list(training_data)
    test_data = list(test_data)

    nn = NeuralNetwork(model=model)
    nn.train(training_data=training_data, evaluation_data=test_data, epochs=epochs)
    
    ask_to_retrain(nn)

# nn = NeuralNetwork.load_parameters(filename="neural_network_parameters.ia")
# print(100 * nn.evaluate(evaluation_data=test_data) / len(test_data), "%")
#
# print(nn.decide(array=test_data[30][0]))
#
# test_data[30][0].resize((28, 28))
# PIL.Image.fromarray((1 - test_data[30][0]) * 255).show()
