from PIL import Image # used for loading images
import numpy as np
import os # used for navigating to image path
import logging
import glob
import re
import pickle
import string
import math
import numpy as np
from tqdm import tqdm
from keras import models
from keras import layers
from keras import optimizers
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
import optparse
import sys
tmp = sys.path
sys.path.append("..")
from common import scores, load_light_dataset, Timer, get_target
sys.path.append(tmp)


parser = optparse.OptionParser()

parser.add_option('-d', '--dataset-dir',
    action="store", dest="dataset_dir",
    help="Directory of the img dataset created with create_images", default="../dataset_img/")
parser.add_option('-t', '--train',
    action="store", dest="train",
    help="true: force training and overwrite the model. false: the trained model will be used", default="false")

options, args = parser.parse_args()


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)

IMG_SIZE = 2640
images_root_dir = options.dataset_dir
levels = 3  # 1=bw 3=rgb
model_name = 'model_trained.k'

target = None
network = {'layers_and_filters': [1, 4], 'kernel_size': [3, 8], 'activation': 'relu', 'learning_rate': 0.001, 'dropout_rate': 0.0, 'optimizer': 'adam', 'epochs': 4}


def load_dataset(root_dir):
    try:
        return pickle.load( open( images_root_dir + "img_dataset.p", "rb" ) )
    except:
        data = np.empty((0,IMG_SIZE**2 * levels + 4), dtype=np.uint8)

        total = 0
        for _ in glob.iglob(root_dir + '**/*.jpeg', recursive=True):
            total += 1

        with tqdm(total=total) as pbar:
            for img_file in glob.iglob(images_root_dir + '**/*.jpeg', recursive=True):
                image = Image.open(img_file, 'r')
                X = np.asarray(image).reshape(-1)[int((2640**2*levels-IMG_SIZE**2*levels)/2):int((2640**2*levels-IMG_SIZE**2*levels)/2+IMG_SIZE**2*levels)]  # immagine ridotta per performace
                Y = get_target(img_file)
                data = np.append(data, np.array([np.append(X,Y)]), axis=0)
                pbar.update(1)

        pickle.dump( data, open( images_root_dir + "img_dataset.p", "wb" ) )
        return data


def prepare_dataset(data, target=3, training_set_part = 0.8):
    '''
    target [0=TRIVIAL,1=STRING,2=REFLECTION,3=CLASS]
    '''
    np.random.shuffle(data)
    X = data[:,:IMG_SIZE**2 * levels]
    Y = data[:,IMG_SIZE**2 * levels + target : IMG_SIZE**2 * levels + target + 1]
    X = X.reshape((-1, IMG_SIZE, IMG_SIZE, levels))

    total_items = X.shape[0]
    input_size = X.shape[1]
    output_size = Y.shape[1]
    training_set_size = int(round(total_items * training_set_part))

    train_X = X[:training_set_size]
    train_Y = Y[:training_set_size]
    test_X = X[training_set_size:]
    test_Y = Y[training_set_size:]

    return train_X, train_Y, test_X, test_Y


def create_model(layers_and_filters, kernels, activation, input_shape, dropout_rate, optimizer, learning_rate, output_size=1):
    model = models.Sequential()
    i = 0
    for filters in layers_and_filters:
        model.add(layers.Conv2D(filters, kernel_size=kernels[i], strides=kernels[i], activation=activation, input_shape=input_shape))
        i += 1
        if i < len(layers_and_filters):
            model.add(layers.MaxPooling2D(pool_size=(2,2)))
            model.add(layers.BatchNormalization())

    if dropout_rate != 0:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(output_size, activation='sigmoid'))

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

    #model.summary()
    return model


def fit_one_at_time(model, files, targets, epochs=1):
    tot = 0
    with tqdm(total=len(files) * epochs) as pbar:
        for _ in range(epochs):
            for i, img_file in enumerate(files):
                try:
                    image = Image.open(img_file, 'r')
                    X = np.asarray(image).reshape((-1, IMG_SIZE, IMG_SIZE, levels))
                    Y = np.array([targets[i]])
                    model.fit(x=X, y=Y, epochs=1, verbose=0)
                except:
                    print('Error: ' + img_file)
                pbar.update(1)


def score_one_at_time(model, files, test_Y):
    preds = np.empty((0,test_Y.shape[1]))

    with tqdm(total=len(files)) as pbar:
        for i, img_file in enumerate(files):
            try:
                image = Image.open(img_file, 'r')
                X = np.asarray(image).reshape((-1, IMG_SIZE, IMG_SIZE, levels))
                Y = model.predict(X, verbose=0)
                preds = np.append(preds, Y, axis=0)
            except:
                print('Error scoring: ' + img_file)
            pbar.update(1)

    print(preds, test_Y)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    print("Other F1 score: ", f1_score(test_Y, preds, average='micro'))

    precision, recall, f1 = scores(preds, test_Y)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f1score: ", f1)


def main_in_memory():
    dataset = load_dataset(images_root_dir)

    train_X, train_Y, test_X, test_Y = prepare_dataset(dataset, target=target)

    model = create_model(
        network['layers_and_filters'],
        network['kernel_size'],
        network['activation'],
        (IMG_SIZE, IMG_SIZE, levels),
        network['dropout_rate'],
        network['optimizer'],
        network['learning_rate'],
        output_size=train_Y.shape[1]
    )
    result = model.fit(
        x=train_X,
        y=train_Y,
        epochs=network['epochs']
        )

    preds = model.predict(test_X)

    print(preds, test_Y)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    print("Other F1 score: ", f1_score(test_Y, preds, average='micro'))
    precision, recall, f1 = scores(preds, test_Y)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1score: ", f1)


def main():
    t = Timer()
    t.reset_cpu_time()
    logging.info("PREPARE DATASET")
    train_X, train_Y, test_X, test_Y = load_light_dataset(images_root_dir, target=target, training_set_part=0.8, extension='jpeg')
    logging.info("CREATE MODEL")
    model = create_model(
        network['layers_and_filters'],
        network['kernel_size'],
        network['activation'],
        (IMG_SIZE, IMG_SIZE, levels),
        network['dropout_rate'],
        network['optimizer'],
        network['learning_rate'],
        output_size=train_Y.shape[1]
    )
    t.get_cpu_time("PREPARATION")
    logging.info("TRAIN")
    try:
        if options.train == 'true':
            raise Exception('Force train model')
        model = models.load_model(model_name)
    except:
        fit_one_at_time(model, train_X, train_Y, epochs=network['epochs'])
        model.save(model_name)
    t.get_cpu_time("TRAIN")
    logging.info("TEST on TRAIN")
    score_one_at_time(model, train_X, train_Y)
    t.get_cpu_time("TEST on TRAIN")
    logging.info("TEST")
    score_one_at_time(model, test_X, test_Y)
    t.get_cpu_time("TEST")


if __name__ == '__main__':
    main()
