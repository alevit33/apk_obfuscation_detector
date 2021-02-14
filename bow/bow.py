import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import losses
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
import logging
import optparse
import sys
tmp = sys.path
sys.path.append("..")
from common import scores, Timer, load_dataset
sys.path.append(tmp)


parser = optparse.OptionParser()

parser.add_option('-d', '--dataset-dir',
    action="store", dest="dataset_dir",
    help="Directory of the text dataset created with count_words", default="../dataset_tfidf.pv")
parser.add_option('-t', '--train',
    action="store", dest="train",
    help="true: force training and overwrite the model. false: the trained model will be used", default="false")

options, args = parser.parse_args()


np.set_printoptions(threshold=sys.maxsize)
logging.basicConfig(level=logging.DEBUG)


network = {'n_layers': 3, 'n_neurons': 50, 'activation': 'sigmoid', 'learning_rate': 0.01, 'dropout_rate': 0.2, 'optimizer': 'adagrad', 'epochs': 10}

target = None
dataset = options.dataset_dir
model_name = 'model_trained.k'


def create_model(input_size, output_size, n_layers, n_neurons, activation_function, learning_rate, dropout_rate, optimizer):
    model = models.Sequential()
    model.add(layers.Dense(n_neurons, input_shape=(input_size, )))
    for _ in range(n_layers):
        if dropout_rate != 0.0:
            model.add(layers.Dropout(dropout_rate, noise_shape=None, seed=None))
        model.add(layers.Dense(n_neurons, activation = activation_function))
    model.add(layers.Dense(output_size, activation = "sigmoid"))
    #model.summary()
    if optimizer == 'rmsprop':
        opt = optimizers.rmsprop(lr=learning_rate)
    elif optimizer == 'adam':
        opt = optimizers.adam(lr=learning_rate)
    elif optimizer == 'sgd':
        opt = optimizers.sgd(lr=learning_rate)
    elif optimizer == 'adagrad':
        opt = optimizers.adagrad(lr=learning_rate)
    elif optimizer == 'adadelta':
        opt = optimizers.adadelta(lr=learning_rate)
    elif optimizer == 'adamax':
        opt = optimizers.adamax(lr=learning_rate)
    elif optimizer == 'nadam':
        opt = optimizers.nadam(lr=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics = ["mean_squared_error"]
    )
    return model


def test(model, test_X, test_Y):
    preds = model.predict(test_X)
    #print(preds)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    print("Other F1 score: ", f1_score(test_Y, preds, average='micro'))
    precision, recall, f1 = scores(preds, test_Y)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f1score: ", f1)


def main():
    t = Timer()
    t.reset_cpu_time()

    # Create traing and test set
    train_X, train_Y, test_X, test_Y = load_dataset(dataset, target=target)

    #dataset beging with the file name
    train_X = train_X[:,1:]
    test_X = test_X[:,1:]

    train_size = train_X.shape[0]
    test_size = test_X.shape[0]
    input_size = train_X.shape[1]
    output_size = train_Y.shape[1]


    try:
        if options.train == 'true':
            raise Exception('Force train model')
        model = models.load_model(model_name)
    except:
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

        t.get_cpu_time("PREPARATION")

        # train the model
        results = model.fit(
         x=train_X,
         y=train_Y,
         epochs= network['epochs'],
         #validation_data= (test_X, test_Y)
        )

        model.save(model_name)

    logging.info("TEST on TRAIN")
    t.reset_cpu_time()
    test(model, np.asarray(train_X).astype(np.float32), train_Y)
    t.get_cpu_time("TEST on TRAIN")
    logging.info("TEST")
    test(model, np.asarray(test_X).astype(np.float32), test_Y)
    t.get_cpu_time("TEST")


if __name__ == '__main__':
    main()
