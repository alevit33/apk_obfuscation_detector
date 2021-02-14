"""Class that represents the network to be evolved."""
import random
import logging

class Network():
    """Represent a network and let us operate on it.
    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices, train_and_score):
        """Initialize our network.
        Args:
            nn_param_choices (dict): Parameters for the network
        """
        self.score = 0.
        self.nn_param_choices = nn_param_choices
        self.train_and_score = train_and_score
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.
        Args:
            network (dict): The network parameters
        """
        self.network = network

    def train(self):
        """Train the network and record the mse.
        Args:
            dataset (str): Name of dataset to use.
        """
        if self.score == 0.:
            self.score = self.train_and_score(self.network)

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network score: %f" % (self.score))