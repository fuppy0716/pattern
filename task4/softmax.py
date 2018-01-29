#!usr/bin/python
# coding: UTF-8
import numpy as np
import math
from mnist import Mnist

class Softmax:

    def __init__(self):
        self.mnist = Mnist()
        self.weight = np.random.normal(0, 0.1, (784, 10))
        self.bias = np.zeros((1,10))
        self.learning_rate = 0.01

        
    def softmax_func(self, u):
        m = np.max(u)
        res = np.exp(u - m) / np.sum(np.exp(u - m))
        return res

    def updata_weight(self):
        for x, l in zip(self.mnist.train_data, self.mnist.train_labels):
            x.shape = (784, 1)
            t = np.zeros((1, 10))
            t[0][l] = 1
            p = self.softmax_func(np.dot(np.transpose(self.weight), x) + np.transpose(self.bias))
            self.weight -= self.learning_rate * np.dot(x, np.transpose(p) - t)
            self.bias -= self.learning_rate * (np.transpose(p) - t)
        return

    def solve(self):
        X = self.mnist.test_data
        W = self.weight
        correct = 0.0
        for x, l in zip(self.mnist.test_data, self.mnist.test_labels):
            u = np.dot(np.transpose(self.weight), x) + self.bias
            p = self.softmax_func(u)
            if (np.argmax(p) == l):
                correct += 1.0
        print correct / self.mnist.test_num
        return correct / self.mnist.test_num


a = Softmax()
print ("0 : "),
a.solve()
for unko in range(1, 11):
    print ("{0} : ".format(unko)),
    a.updata_weight()
    a.solve()
