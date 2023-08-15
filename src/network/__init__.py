# Preceding underscore to assure that this is imported first
from ._neural_network import NeuralNetwork
from .evaluation import evaluate_network
from .feed_forward_network import FeedForwardNetwork

__all__ = [
    "FeedForwardNetwork",
    "NeuralNetwork",
    "evaluate_network",
]

__author__ = ["Roäc Ravenhill"]
