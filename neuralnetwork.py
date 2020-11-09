
# This script was written by Mabrur Rahman (2020). Last modified November 2020
# 
# This scipt holds the Neural Network classes. The base network is modified 
# form of the network written by Tariq Rashid (2016) from his book, Make Your 
# Own Neural Network. 
# 
# Each sub-class is a network specifically designed to deal with a certain 
# problem. The MNIST neural network is specifically designed for the MNIST 
# numbers dataset, and the Dimensions Network is specifically designed for my 
# project


import numpy as np
import scipy.special
from timeit import default_timer as timer


class Network:
    """
    A beginner-level neural network.\n
    Establishes a neural network with variable input, hidden and output nodes, 
    as well as variable hidden layers.
    """


    def __init__(self, input_nodes, hidden_nodes, output_nodes, activ_func, learning_rate):
        """
        Initialises the network.\n
        Requries number of input, hidden (as a list), and output nodes, the activation function ('sigmoid' or 'relu'), and the learning rate
        """
        
        # Number of nodes in each layer
        self.in_nodes = input_nodes
        self.hid_nodes = hidden_nodes   # Hidden nodes are given as a list
        self.out_nodes = output_nodes

        # Number of hidden layers. The length of the list indicates how many 
        # hidden layers are in the network
        self.hid_layers = len(self.hid_nodes)

        # Learning rate
        self.lr = learning_rate

        # Activation function is defined as specified
        self.activ_func = activ_func

        if self.activ_func == "sigmoid":
            self.activation_func = lambda x: scipy.special.expit(x)
        elif self.activ_func == "relu":
            self.activation_func = lambda x: (x > 0) * x

        # Establish a matrix of weights, and append the weights of each layer 
        # into a list. Using a list allows for varying hidden layers with 
        # varying nodes
        self.weights = [np.random.normal(0.0, pow(self.hid_nodes[0], -0.5), (self.hid_nodes[0], self.in_nodes))]
        
        for layer in range(1, self.hid_layers):
            self.weights.append(np.random.normal(0.0, pow(self.hid_nodes[layer], -0.5), (self.hid_nodes[layer], self.hid_nodes[layer - 1])))
        
        self.weights.append(np.random.normal(0.0, pow(self.out_nodes, -0.5), (self.out_nodes, self.hid_nodes[-1])))


    def networkStatus(self):
        """
        The status of the neural network. Checks to see if the weights have been calculated properly.
        """

        # Each attribute is calculated based on the corresponding weight matrix size. This is a good way to ensure that the weights were calculated properly
        print(f"Input nodes: \t\t{np.size(self.weights[0], axis=1)}")
        print(f"Hidden layers: \t\t{len(self.weights) - 1}")
        print(f"Hidden nodes: \t\t{[np.size(self.weights[i], axis=0) for i in range(0, self.hid_layers)]}")
        print(f"Output nodes: \t\t{np.size(self.weights[-1], axis=0)}")
        print(f"Activation function: \t{self.activ_func}")
        print(f"Learning rate: \t\t{self.lr}")


    def trainNetwork(self, inputs_list, targets_list):
        """
        Trains the neural network using back propgation.\n
        Requries a matrix of inputs values, and a matrix of target values
        """

        # Convert the input and target lists into a 2D array. Transpose the input lists from a 1xn matrix to a nx1 matrix
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Query the network, and store the values from each layer into a list. Using a list allows for varying hidden layers
        outputs = [self.activation_func(np.dot(self.weights[0], inputs))]
        
        for layer in range(1, self.hid_layers + 1):
            outputs.append(self.activation_func(np.dot(self.weights[layer], outputs[layer - 1])))

        # Calculate the error value from each layer, starting from the final output layers, and update the corresponding weights using back propagation
        next_error = targets - outputs[-1]  # The outer layer
        if self.activ_func == "sigmoid":
            
            for layer in range(self.hid_layers, 0, -1):
                
                # Update per layer
                self.weights[layer] += self.lr * np.dot((next_error * outputs[layer] * (1.0 - outputs[layer])), np.transpose(outputs[layer - 1]))   
                
                # Calculate per layer
                prev_error = next_error
                next_error = np.dot(self.weights[layer].T, prev_error)  

            # Update the input layer
            self.weights[0] += self.lr * np.dot((next_error * outputs[0] * (1.0 - outputs[0])), np.transpose(inputs))   

        elif self.activ_func == "relu":

            for layer in range(self.hid_layers, 0, -1):
                
                # Update per layer
                self.weights[layer] += self.lr * np.dot((next_error * (outputs[layer] > 0) * 1.0), np.transpose(outputs[layer - 1]))   
                
                # Calculate per layer
                prev_error = next_error
                next_error = np.dot(self.weights[layer].T, prev_error)  

            # Update the input layer
            self.weights[0] += self.lr * np.dot((next_error * (outputs[0] > 0) * 1.0), np.transpose(inputs))   


    def queryNetwork(self, inputs_list):
        """
        Performs a forward propagation of the neural network.\n
        Requires a matrix of input values.
        """
        
        # Convert the input and target lists into a 2D array. Transpose the input lists from a 1xn matrix to a nx1 matrix
        inputs = np.array(inputs_list, ndmin=2).T

        # Query the network
        final_output = self.activation_func(np.dot(self.weights[0], inputs))
        for layer in range(1, self.hid_layers + 1):
            prev_output = final_output
            final_output = self.activation_func(np.dot(self.weights[layer], prev_output))

        return final_output



class MNISTNetwork(Network):
    """
    A neural network specifically for the MNIST handwriting recognition task.\n
    This class sets default values for the amount of input nodes (784, as per the 28x28 image size) and output nodes (10). 
    The number of hidden nodes (and layers) is set to 1 layer of 100 nodes ([100]) by default which can be changed.
    The learning rate is set to 0.3 by default which can be changed.
    """

    INPUT_NODES = 784
    OUTPUT_NODES = 10


    def __init__(self, hid_nodes=[100], activ_func="sigmoid", lr=0.3):
        super().__init__(MNISTNetwork.INPUT_NODES, hid_nodes, MNISTNetwork.OUTPUT_NODES, activ_func, lr)
    

    def train(self, train_list, epochs=1):
        """
        Training the neural network using back propagation.\n 
        Requires raw training data. The data will be prepared before training the network.
        The number of epochs is set to 1 by default which can be changed.
        """

        start = timer()

        # Preparing the data for training
        for e in range(epochs):
            for record in train_list:
                all_values = record.split(',')

                # Scale and shift the inputs
                inputs = (np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01

                # Create target output values
                targets = np.zeros(MNISTNetwork.OUTPUT_NODES) + 0.01
                targets[int(all_values[0])] = 0.99      # all_values[0] is the target label

                super().trainNetwork(inputs, targets)
        
        end = timer()
        
        print(f"The network has trained {e+1} time(s).")
        print(f"Training took {(end - start) / 60:.0f} minutes and {(end - start) % 60:.2f} seconds.")


    def test(self, test_list):
        """
        Tests the neural network.\n 
        Requires raw testing data. The data will be prepared before testing the network.
        """

        scorecard = []
        for record in test_list:
            all_values = record.split(',')
            
            # Correct answer is the first value
            correct_label = int(all_values[0])
            
            # Scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # Query the network
            outputs = super().queryNetwork(inputs)

            # The index of the highest value corresponds to the label
            label = np.argmax(outputs)
          
            # Append correct or incorrect to the scorecard
            if label == correct_label:
                scorecard.append(1)
            else:
                scorecard.append(0)
                # print(f"Was {correct_label}, thought it was {label}")
        
        # Calculate the performance of the network
        scorecard_array = np.asarray(scorecard)
        print(f"\nPerformance = {(scorecard_array.sum() / scorecard_array.size) * 100.0:.2f}%")  


    
class DimensionsNetwork(Network):
    """
    A neural network specifically for the my project.\n
    This class sets default values for the amount of input nodes (784, as per the 28x28 image size) and output nodes (4). 
    The number of hidden nodes (and layers) is set to 2 layer of 100 nodes ([100, 100]) by default which can be changed.
    The activation function is set to sigmoid by default which can be changed (to ReLU).
    The learning rate is set to 0.1 by default which can be changed.
    """

    INPUT_NODES = 784
    OUTPUT_NODES = 4


    def __init__(self, hid_nodes=[100, 100], activ_func="sigmoid", lr=0.1):
        super().__init__(DimensionsNetwork.INPUT_NODES, hid_nodes, DimensionsNetwork.OUTPUT_NODES, activ_func, lr)

        self.activ = activ_func
    

    def train(self, train_list, epochs=1):
        """
        Training the neural network using back propagation.\n 
        Requires raw training data. The data will be prepared before training the network.
        The number of epochs is set to 1 by default which can be changed.
        """

        start = timer()

        # Preparing the data for training
        for e in range(epochs):
            for record in train_list:
                all_values = record.split(',')

                # Scale and shift the inputs
                inputs = (np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01

                # Create target output values
                targets = np.zeros(DimensionsNetwork.OUTPUT_NODES) + 0.01
                targets[int(float(all_values[0]))] = 0.99

                super().trainNetwork(inputs, targets)
        
        end = timer()
        
        print(f"The network has trained {e+1} time(s).")
        print(f"Training took {(end - start) / 60:.0f} minutes and {(end - start) % 60:.2f} seconds.\n")
        
        # Print out information about the dataset
        self.__dataset('train', train_list)
        print("\n\n")


    def test(self, test_list):
        """
        Tests the neural network.\n 
        Requires raw testing data. The data will be prepared before testing the network.
        """

        scorecard = []
        for record in test_list:
            all_values = record.split(',')
            
            # Correct answer is the first value
            correct_label = int(float(all_values[0]))
            
            # Scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            # Query the network
            outputs = super().queryNetwork(inputs)

            # The sum of the indicies of the 2 highest value corresponds to the label
            ind = np.argpartition(outputs, -3, axis=0)[-2:]
            label = np.sum(ind)
            
            # Append correct or incorrect to the scorecard
            if label == correct_label:
                scorecard.append(1)
            else:
                scorecard.append(0)
        
        # Print information about the datalist
        self.__dataset('test', test_list)
        
        # Calculate the performance of the network
        scorecard_array = np.asarray(scorecard)
        print(f"\nPerformance ({self.activ}) = {(scorecard_array.sum() / scorecard_array.size) * 100.0:.2f}%")  
        print("\n\n")


    def train_over_time(self, train_list, test_list, epochs=1):
        """
        This function will train and test the network over the specified epochs.\n
        Requires both raw training and test data. The data will be prepared before training the network.
        The number of epochs is set to 1 by default which can be changed.
        """
        
        # Print information about both datasets
        self.__dataset("train", train_list)
        print("\n")
        self.__dataset("test", test_list)
        print("\n")
        
        print("-------------------------------------------")
        print(f"Performance data: {self.activ}\n")


        start = timer()

        for e in range(epochs):

            # Begin training
            for record in train_list:
                all_values = record.split(',')

                # Scale and shift the inputs
                inputs = (np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01

                # Create target output values
                targets = np.zeros(DimensionsNetwork.OUTPUT_NODES) + 0.01
                targets[int(float(all_values[0]))] = 0.99

                super().trainNetwork(inputs, targets)

            # Begin testing
            scorecard = []
            for record in test_list:
                all_values = record.split(',')
                
                # Correct answer is the first value
                correct_label = int(float(all_values[0]))
                
                # Scale and shift the inputs
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

                # Query the network
                outputs = super().queryNetwork(inputs)

                # The sum of the indicies of the 2 highest value corresponds to the label
                ind = np.argpartition(outputs, -3, axis=0)[-2:]
                label = np.sum(ind)

                # Append correct or incorrect to the scorecard
                if label == correct_label:
                    scorecard.append(1)
                else:
                    scorecard.append(0) 
            
            # Calculate and print performance before training again
            scorecard_array = np.asarray(scorecard)
            print(f"Performance after {e+1:2} epochs = {(scorecard_array.sum() / scorecard_array.size) * 100.0:.2f}%")  
           
        end = timer()
        
        print("\n")
        print(f"The network has trained {e+1} time(s).")
        print(f"Training took {(end - start) / 60:.0f} minutes and {(end - start) % 60:.2f} seconds.\n")


    def __dataset(self, action, data_list):
        """
        This function will calculate and print information about the specified dataset.\n
        Requires an action ('train' or 'test') and the corresponding dataset.
        Each dataset is hardcoded, meaning that any other dataset will not work.
        """

        # Initialise a dictionary that holds each label
        if action == 'train':
            data = {0: 0, 
                    1: 0, 
                    2: 0,  
                    3: 0}
        elif action == 'test':
            data = {1: 0, 
                    2: 0, 
                    4: 0, 
                    5: 0}
        
        # Count how many stimuli of each label exists in the dataset
        for record in data_list:
            data[int(record[0])] += 1
        
        total = sum(list(data.values()))

        # Print the relevant information
        if action == 'train':
            print(f"Trained on {total} datapoints.")
            print(f"Small square and down diagonal (\):\t{data[0] / total * 100:.2f}%")
            print(f"Small square and up diagonal (/):\t{data[1] / total * 100:.2f}%")
            print(f"Large square and down diagonal (\):\t{data[2] / total * 100:.2f}%")
            print(f"Large square and up diagonal (/):\t{data[3] / total * 100:.2f}%")
        elif action == 'test':
            print(f"Tested on {total} datapoints.")
            print(f"Small square:\t\t{data[1] / total * 100:.2f}%")
            print(f"Large square:\t\t{data[5] / total * 100:.2f}%")
            print(f"Down diagonal (\):\t{data[2] / total * 100:.2f}%")
            print(f"Up diagonal (/):\t{data[4] / total * 100:.2f}%")