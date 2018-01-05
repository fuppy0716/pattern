#!usr/bin/python
# coding: UTF-8
import numpy as np
import math

class Functions:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z):
        mx = np.max(z)
        return np.exp(z - mx) / np.sum(np.exp(z - mx))

    @staticmethod
    def relu(z):
        return np.maximum(z, 0)
