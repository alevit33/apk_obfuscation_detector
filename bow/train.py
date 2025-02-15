"""
Utility used by the Network class to actually train.
Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import losses
from keras import optimizers
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
import logging
from bow import create_model
tmp = sys.path
sys.path.append("..")
from common import scores, load_dataset
sys.path.append(tmp)



target = None   # [0=TRIVIAL,1=STRING,2=REFLECTION,3=CLASS]

train_X, train_Y, test_X, test_Y = load_dataset("../dataset_tfidf.pv", target=target);

#dataset beging with the file name
train_X = train_X[:,1:]
test_X = test_X[:,1:]

train_size = train_X.shape[0]
test_size = test_X.shape[0]
input_size = train_X.shape[1]
output_size = train_Y.shape[1]

def train_and_score(network):
    """Train the model, return test loss.
    Args:
        network (dict): the parameters of the network
    """

    # create the model
    model = create_model(
        input_size=input_size,
        output_size=output_size,
        n_layers=network['n_layers'],
        n_neurons=network['n_neurons'],
        activation_function=network['activation'],
        learning_rate=network['learning_rate'],
        dropout_rate=network['dropout_rate'],
        optimizer=network['optimizer']
    )

    # train the model
    results = model.fit(
     x=train_X,
     y=train_Y,
     epochs= network['epochs']
    )

    preds = model.predict(test_X)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    if len(preds) == len(preds[preds == 0]):
        f1_score = 0.0
    else:
        _, _, f1_score = scores(preds, test_Y)
    return np.mean(f1_score)
