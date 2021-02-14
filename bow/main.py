import pickle
from math import sqrt

import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from matplotlib import pyplot
from numpy import array
from pandas import DataFrame
from pandas import Series
from pandas import concat


import logging
from optimizer import Optimizer
from train import train_and_score
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='results/training_log_13-adhoc-tfidf.txt'
)

logging.info("----------------------DATASET ALL-----------------------")

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
    optimizer = Optimizer(nn_param_choices, train_and_score)
    networks = optimizer.create_population(population)

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

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.score, reverse=True)

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
    """Evolve a network."""
    generations = 5  # Number of times to evole the population.
    population = 16  # Number of networks in each generation.

    nn_param_choices = {
        #'pv_size': [50,100,200],
        #'wv_size': [30,50,100],
        #'min_freq_word': [3,6,10,20],
        #'network_type': ['deep'],
        'n_layers': [0, 1, 2, 3, 4],
        'n_neurons': [20, 50, 100, 300, 500],
        'activation': ['relu', 'tanh', 'sigmoid', 'linear'],
        'learning_rate': [1.0, 0.1, 0.05, 0.01, 0.001],
        'dropout_rate': [0.0, 0.1, 0.2],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
        'epochs': [5]#[1,3,10,30]
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices)

if __name__ == '__main__':
    main()



