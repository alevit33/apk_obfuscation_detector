from PIL import Image # used for loading images
import numpy as np
import os # used for navigating to image path
import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='cnn_training_log.txt'
)
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
from math import sqrt
from cnn import create_model, load_dataset, prepare_dataset, fit_one_at_time
import optparse
import sys
tmp = sys.path
sys.path.append("..")
from common import scores, load_light_dataset, Timer, get_target
from optimizer import Optimizer
sys.path.append(tmp)


parser = optparse.OptionParser()

parser.add_option('-d', '--dataset-dir',
    action="store", dest="dataset_dir",
    help="Directory of the img dataset created with create_images", default="../dataset_img/")

options, args = parser.parse_args()


target = 2
targets = ["TRIVIAL" , "STRING", "REFLECTION", "CLASS", "ALL"]

IMG_SIZE = 2640
images_root_dir = options.dataset_dir
levels = 3  # 1=bw 3=rgb


def score_one_at_time(model, files, test_Y):
    preds = np.empty((0,test_Y.shape[1]))
    
    with tqdm(total=len(files)) as pbar:
        for i, img_file in enumerate(files):
            image = Image.open(img_file, 'r')
            X = np.asarray(image).reshape((-1, IMG_SIZE, IMG_SIZE, levels))
            Y = model.predict(X, verbose=0)
            preds = np.append(preds, Y, axis=0)
            pbar.update(1)
    
    
    std_dev = np.std(preds)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    _, _, f1 = scores(preds, test_Y)
    return np.mean(f1) + 2 * std_dev  # std deviation to avoid results near 0.5 or all equals


train_X, train_Y, test_X, test_Y = load_light_dataset(images_root_dir, target=target, training_set_part=0.8, extension='jpeg')
#dataset = load_dataset(images_root_dir)
#train_X, train_Y, test_X, test_Y = prepare_dataset(dataset, target=target)

def train_and_score(network):
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
    fit_one_at_time(model, train_X, train_Y, epochs=network['epochs'])
    return score_one_at_time(model, test_X, test_Y)


def train_and_score_in_memory(network):
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
    std_dev = np.std(preds)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    _, _, f1 = scores(preds, test_Y)
    return np.mean(f1) + 2 * std_dev





def train_networks(networks):
    """Train each network.
    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train()
        pbar.update(1)
    pbar.close()

def get_average_score(networks):
    """Get the average score for a group of networks.
    Args:
        networks (list): List of networks
    Returns:
        float: The average score of a population of networks.
    """
    total_score = 0
    for network in networks:
        total_score += network.score

    return total_score / len(networks)

def generate(generations, population, nn_param_choices):
    """Generate a network with the genetic algorithm.
    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating
    """
    optimizer = Optimizer(nn_param_choices, train_and_score, retain=0.5, random_select=0.4)
    networks = optimizer.create_population(population)
    all_networks = []

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get score for networks.
        train_networks(networks)

        # Get the average score for this generation.
        average_score = get_average_score(networks)

        # Print out the average score each generation.
        logging.info("Generation F1-score average: %.2f" % (average_score))
        logging.info('-'*80)
        
        all_networks = all_networks + networks

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(all_networks, key=lambda x: x.score, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
    """Print a list of networks.
    Args:
        networks (list): The population of networks
    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main():
    logging.info("----------------------TARGET: " + targets[target] + "-----------------------")
    
    """Evolve a network."""
    generations = 3  # Number of times to evole the population.
    population = 5  # Number of networks in each generation.

    nn_param_choices = {
        'layers_and_filters': [[1],[4],[2],[1,4],[1,2],[4,1],[1,2,1],[1,2,2]],
        'kernel_size': [[3,8,4],[3,3,4],[8,4,4],[4,8,4],[3,16,4]],
        'activation': ['relu', 'tanh', 'sigmoid', 'linear'],
        'learning_rate': [0.1, 0.01, 0.001],
        'dropout_rate': [0, 0.1, 0.2],
        'optimizer': ['adam', 'sgd', 'adamax'],
        'epochs': [1]
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices)

if __name__ == '__main__':
    main()


