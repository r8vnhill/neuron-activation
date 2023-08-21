# Preceding underscore to assure that this is imported first
from ._neural_network import NeuralNetwork
from .feed_forward_network import FeedForwardNetwork

__all__ = [
    "FeedForwardNetwork",
    "NeuralNetwork",
]

__author__ = ["Ignacio Slater"]
