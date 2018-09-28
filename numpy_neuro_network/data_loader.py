import pandas as pd
import numpy as np               #for maths

def dataLoader(path):
    dataset = pd.read_csv(path)
    dataset = pd.get_dummies(dataset, columns=['species'])
    values = list(dataset.columns.values)
    y = dataset[values[-3:]]
    y = np.array(y, dtype='int32')
    X = dataset[values[:4]]
    X = np.array(X, dtype='float32')
    num_labels = X[0]
    target = []
    for i in range(len(y)):
        if np.array_equal(y[i], [1, 0, 0]):
            target.append(0)
        elif np.array_equal(y[i], [0, 1, 0]):
            target.append(1)
        elif np.array_equal(y[i], [0, 0, 1]):
            target.append(2)

    target = np.asarray(target, dtype='int32')
    return X, target, num_labels