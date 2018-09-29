# Numpy Neural Network
Neural Network from scratch in Python exclusively using Numpy.
## Overview

This project consists of a neural network implementation from scratch. Modules are organized in a way that intends to provide both an understandable implementation of neural networks and a user-friendly API.
The project is structured as follows:
- `numpy_neuro_network`
    - data_loader.py: load iris dataset from ../dataset/iris.csv
	- run_2_hidden_layer.py: main entrance of this project.
## Prerequisites

- Python 3.5
- Pip 9.0.1

Note: Not tested with other python versions.

## Installation
Once you have met the prerequisites, a single step is required to install this software:
1. Run `python setup.py install`

This will install `numpy` (the only required external library to run the neural network) and `matplotlib` (only needed to plot classifier boundaries when running an example).

## Further improvements

There are several functionalities that may be implemented to make this software more useful:
- Other types of layers: LSTM, CNN, embeddings,...
- Batches
- More optimizers other than Stochastic Gradient Descent
- More activations
- More initializers
- More objective functions
- Regularization
- Parallelization
