#!usr/bin/python
# coding: UTF-8
import numpy as np
import math
import gzip
import os.path

class Mnist:

    key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
    }
    dataset_dir = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.train_num = 60000
        self.test_num = 10000
        self.img_dim = (1, 28, 28)
        self.img_size = 784
        datas = self.load_mnist()
        self.train_data = datas[0]
        self.train_labels = datas[1]
        self.test_data = datas[2]
        self.test_labels = datas[3]

    def load_img(self, filename):
        file_path = Mnist.dataset_dir + '/' + filename
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data.flags.writeable = True
        return data.reshape(-1, self.img_size)

    def load_label(self, filename):
        file_path = Mnist.dataset_dir + '/' + filename
        with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels
    
    def load_mnist(self):
        train_data = self.load_img(Mnist.key_file['train_img'])
        train_labels = self.load_label(Mnist.key_file['train_label'])
        test_data = self.load_img(Mnist.key_file['test_img'])
        test_labels = self.load_label(Mnist.key_file['test_label'])
        train_data = train_data / 255.0
        test_data = test_data / 255.0
        return [train_data, train_labels, test_data, test_labels]

