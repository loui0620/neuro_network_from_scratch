import numpy as np               #for maths
import matplotlib.pyplot as plt  #for plotting
import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from numpy_neuro_network.data_loader import dataLoader

#HYPERPARAMETERS
#THis network is aiming for classification of Iris dataset.

#define layer_neurons
input_units  = 4   #neurons in input layer
hidden1_units = 8   #neurons in 1st hidden layer
hidden2_units = 8   #neurons in 2nd hidden layer
output_units = 3   #neurons in output layer

#num of target labels
num_classes = output_units

#define hyper-parameters
learning_rate = 0.02

#regularization parameter
beta = 0.0001

#num of iterations
iters = 5001


# initialize parameters i.e weights
def initialize_parameters():
    # initial values should have zero mean and 0.1 standard deviation
    mean = 0  # mean of parameters
    std = 0.03  # standard deviation

    layer1_weights = np.random.normal(mean, std, (input_units, hidden1_units))
    layer1_biases = np.ones((hidden1_units, 1))

    layer2_weights = np.random.normal(mean, std, (hidden1_units, hidden2_units))
    layer2_biases = np.ones((hidden2_units, 1))

    layer3_weights = np.random.normal(mean, std, (hidden2_units, output_units))
    layer3_biases = np.ones((output_units, 1))

    parameters = dict()
    parameters['layer1_weights'] = layer1_weights
    parameters['layer1_biases'] = layer1_biases
    parameters['layer2_weights'] = layer2_weights
    parameters['layer2_biases'] = layer2_biases
    parameters['layer3_weights'] = layer3_weights
    parameters['layer3_biases'] = layer3_biases

    return parameters

#activation function
def sigmoid(X):
    return 1/(1+np.exp((-1)*X))

#softmax function for output
def softmax(X):
    exp_X = np.exp(X)
    exp_X_sum = np.sum(exp_X,axis=1).reshape(-1,1)
    exp_X = (exp_X/exp_X_sum)
    return exp_X

def ReLU(X):
    return abs(X) * (X > 0)


def forward_propagation(train_dataset, parameters):
    cache = dict()  # to store the intermediate values for backward propagation
    m = len(train_dataset)  # number of training examples

    # get the parameters
    layer1_weights = parameters['layer1_weights']
    layer1_biases = parameters['layer1_biases']
    layer2_weights = parameters['layer2_weights']
    layer2_biases = parameters['layer2_biases']
    layer3_weights = parameters['layer3_weights']
    layer3_biases = parameters['layer3_biases']

    # forward prop
    logits = np.matmul(train_dataset, layer1_weights) + layer1_biases.T
    activation1 = np.array(sigmoid(logits)).reshape(m, hidden1_units)
    activation2 = np.array(np.matmul(activation1, layer2_weights) + layer2_biases.T).reshape(m, hidden2_units)
    activation3 = np.array(np.matmul(activation2, layer3_weights) + layer3_biases.T).reshape(m, output_units)
    output = np.array(softmax(activation3)).reshape(m, num_classes)

    # fill in the cache
    cache['output'] = output
    cache['activation1'] = activation1
    cache['activation2'] = activation2
    cache['activation3'] = activation3

    return cache, output


# backward propagation
def backward_propagation(train_dataset, train_labels, parameters, cache):
    derivatives = dict()  # to store the derivatives

    # get stuff from cache
    output = cache['output']
    activation1 = cache['activation1']
    activation2 = cache['activation2']

    # get parameters
    layer1_weights = parameters['layer1_weights']
    layer2_weights = parameters['layer2_weights']
    layer3_weights = parameters['layer3_weights']

    # calculate errors
    error_output = output - train_labels
    error_activation2 = np.matmul(error_output, layer3_weights.T)
    error_activation2 = np.multiply(error_activation2, activation2)
    error_activation2 = np.multiply(error_activation2, 1 - activation2)

    error_activation1 = np.matmul(error_activation2, layer2_weights.T)
    error_activation1 = np.multiply(error_activation1, activation1)
    error_activation1 = np.multiply(error_activation1, 1-activation1)

    # calculate partial derivatives
    partial_derivatives3 = np.matmul(activation2.T, error_output) / len(train_dataset)
    partial_derivatives2 = np.matmul(activation1.T, error_activation2) / len(train_dataset)
    partial_derivatives1 = np.matmul(train_dataset.T, error_activation1) / len(train_dataset)

    # store the derivatives
    derivatives['partial_derivatives1'] = partial_derivatives1
    derivatives['partial_derivatives2'] = partial_derivatives2
    derivatives['partial_derivatives3'] = partial_derivatives3

    return derivatives

# update the parameters
def update_parameters(derivatives, parameters):
    # get the parameters
    layer1_weights = parameters['layer1_weights']
    layer2_weights = parameters['layer2_weights']
    layer3_weights = parameters['layer3_weights']

    # get the derivatives
    partial_derivatives1 = derivatives['partial_derivatives1']
    partial_derivatives2 = derivatives['partial_derivatives2']
    partial_derivatives3 = derivatives['partial_derivatives3']

    # update the derivatives
    layer1_weights -= (learning_rate * (partial_derivatives1 + beta * layer1_weights))
    layer2_weights -= (learning_rate * (partial_derivatives2 + beta * layer2_weights))
    layer3_weights -= (learning_rate * (partial_derivatives3 + beta * layer3_weights))

    # update the dict
    parameters['layer1_weights'] = layer1_weights
    parameters['layer2_weights'] = layer2_weights
    parameters['layer3_weights'] = layer3_weights

    return parameters


# calculate the loss and accuracy
def cal_loss_accuray(train_labels, predictions, parameters):
    # get the parameters
    layer1_weights = parameters['layer1_weights']
    layer2_weights = parameters['layer2_weights']
    layer3_weights = parameters['layer3_weights']

    # cal loss and accuracy
    loss = -1 * np.sum(np.multiply(np.log(predictions), train_labels) + np.multiply(np.log(1 - predictions),
                                                                                    (1 - train_labels))) + np.sum(
        layer1_weights ** 2) * beta + np.sum(layer2_weights ** 2) * beta + np.sum(layer3_weights ** 2) * beta
    loss /= len(train_labels)
    accuracy = np.sum(np.argmax(train_labels, axis=1) == np.argmax(predictions, axis=1))
    accuracy /= 150

    return loss, accuracy


# training function
def train(train_dataset, train_labels, iters=2):
    # To store loss after every iteration.
    J = []

    # WEIGHTS
    global layer1_weights, layer1_biases, layer2_weights, layer2_biases, layer3_weights, layer3_biases

    # initialize the parameters
    parameters = initialize_parameters()

    layer1_weights = parameters['layer1_weights']
    layer1_biases = parameters['layer1_biases']
    layer3_weights = parameters['layer3_weights']
    layer3_biases = parameters['layer3_biases']

    # to store final predictons after training
    final_output = []
    for j in range(iters):
        # forward propagation
        cache, output = forward_propagation(train_dataset, parameters)

        # backward propagation
        derivatives = backward_propagation(train_dataset, train_labels, parameters, cache)

        # calculate the loss and accuracy
        loss, accuracy = cal_loss_accuray(train_labels, output, parameters)

        # update the parameters
        parameters = update_parameters(derivatives, parameters)

        # append loss
        J.append(loss)

        # update final output
        final_output = output

        # print accuracy and loss
        if (j % 500 == 0):
            print("Step %d" % j)
            print("Loss %f" % loss)
            print("Accuracy %f%%" % (accuracy * 100))

    return J, final_output

def main():
    data, target, num_labels = dataLoader('../dataset/iris.csv')
    #shuffle the dataset
    z = list(zip(data,target))
    np.random.shuffle(z)
    data,target = zip(*z)

    #make train_dataset and train_labels
    train_dataset = np.array(data).reshape(-1,4)
    train_labels = np.zeros([train_dataset.shape[0],num_classes])

    #one-hot encoding
    for i,label in enumerate(target):
        train_labels[i,label] = 1

    #normalizations
    for i in range(input_units):
        mean = train_dataset[:,i].mean()
        std = train_dataset[:,i].std()
        train_dataset[:,i] = (train_dataset[:,i]-mean)/std

    #train data
    J,final_output = train(train_dataset,train_labels,iters=4001)

    #plot loss graph
    plt.plot(list(range(1,len(J))),J[1:])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Iterations & Loss')
    plt.show()

if __name__ == "__main__":
    main()