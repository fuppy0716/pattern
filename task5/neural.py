#!usr/bin/python
# coding: UTF-8
import numpy as np
import math
from mnist import Mnist
from functions import Functions

class Neural:

    def __init__(self):
        self.mnist = Mnist()
        self.layer_num = 3
        self.shape = [784, 100, 50, 10]
        self.func = [Functions.relu, Functions.relu, Functions.softmax]
        self.learning_rate = 1.0
        self.batch_size = 100
        self.weight = self.initWeight()
        self.bias = self.initBias()

    def initWeight(self):
        weight = [] #weight[i] = W(i + 1)
        for i in range(self.layer_num):
            weight.append(np.random.normal(0, 0.1, (self.shape[i], self.shape[i + 1])))
        return weight

    def initBias(self):
        bias = [] #bias[i] = b(i + 1)
        for i in range(self.layer_num):
            bias.append(np.zeros((self.shape[i + 1], 1)))
        return bias

    def forwardPropagation(self, z0):
        U = [] #U[i] = u(i+1)資料の添字
        Z = []
        z = np.reshape(z0, (784, 1))  
        Z.append(z) #Z[i] = z(i)
        for i in range(self.layer_num):
            u = np.dot(self.weight[i].T, z) + self.bias[i]
            U.append(u)
            z = self.func[i](u)
            Z.append(z)
        return [U, Z]

    def backPropagation(self, U, Z, deltaWeight, deltaBias, t):
        t = np.reshape(t, (10, 1))
        delta = Z[self.layer_num] - t
        deltaWeight[self.layer_num - 1] = deltaWeight[self.layer_num - 1] + np.dot(Z[self.layer_num - 1], delta.T)
        deltaBias[self.layer_num - 1] = deltaBias[self.layer_num - 1] + delta
        for l in range(self.layer_num - 1, 0, -1):
            delta = Z[l]*(np.dot(self.weight[l], delta))
            deltaWeight[l - 1] = deltaWeight[l - 1] + np.dot(Z[l - 1], delta.T)
            deltaBias[l - 1] = deltaBias[l - 1] + delta
        return [deltaWeight, deltaBias]

    def updateWeight(self):
        for m in range(self.mnist.train_num / self.batch_size * 20): #20エポック
            if (m % (self.mnist.train_num / self.batch_size) == 0):
                self.showAccuracy()

            if (m % (10 * self.mnist.train_num / self.batch_size) == 0):
                print "aaa"
                self.learning_rate /= 10.0
                
            batch_mask = np.random.choice(self.mnist.train_num, self.batch_size, replace=False)
            X = self.mnist.train_data[batch_mask]
            T = self.mnist.train_labels[batch_mask]
            deltaWeight = []
            deltaBias = []
            for l in range(self.layer_num):
                deltaWeight.append(np.zeros(self.weight[l].shape))
                deltaBias.append(np.zeros(self.bias[l].shape))

            for i in range(self.batch_size):
                res = self.forwardPropagation(X[i])
                U = res[0]; Z = res[1]
                res = self.backPropagation(U, Z, deltaWeight, deltaBias, T[i])
                deltaWeight = res[0]; deltaBias = res[1]

            for l in range(self.layer_num):
                self.weight[l] = self.weight[l] - (self.learning_rate / self.batch_size) * deltaWeight[l]
                self.bias[l] = self.bias[l] - (self.learning_rate / self.batch_size) * deltaBias[l]
        return

    def showAccuracy(self):
        X = self.mnist.test_data
        T = self.mnist.test_labels
        cnt = 0
        for i in range(self.mnist.test_num):
            x = X[i]
            res = self.forwardPropagation(x)
            Z = res[1]
            if np.argmax(Z[self.layer_num]) == np.argmax(T[i]):
                cnt = cnt + 1
        print cnt
        return
    
a = Neural()
print "batchsize : ",
print a.batch_size
a.updateWeight()
a.showAccuracy()
