#!usr/bin/python
# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math

i0 = 5
i1 = 2
d = 3


def read_file():    
    columns = ["mpg", "cylinders", "displacement", "horsepower", "weight",
            "acceleration", "modelyear", "origin", "carname"]

    df = pd.read_csv('auto-mpg.csv', names = columns, header = None)
    df = df.query("horsepower != '?'")
    df.index = range(len(df))
    df[["horsepower"]] = df[["horsepower"]].astype(float)
    return df



def analysis(data, columns):
    x1 = np.array([data[columns[i0]]])
    x2 = np.array([data[columns[i1]]])
    t = np.array([data[columns[d]]])
    t = np.transpose(t)
    x = np.vstack((x1,x2))
    x = np.transpose(x)
    ans = np.dot(np.transpose(x),x)
    ans = np.dot(np.linalg.inv(ans),np.transpose(x))
    ans = np.dot(ans,t)
    return ans

def func(w,X1,X2):
    y = np.eye(len(X1),len(X1[0]))
    for i in range(len(y)):
        for j in range(len(y[0])):
            x = np.array([[X1[i][j]],[X2[i][j]]])
            temp = np.dot(np.transpose(w),x)
            y[i][j] = temp[0][0]
    return y

def plot(w, x1, x2, t):
    x1m = np.min(x1)
    x1M = np.max(x1)
    x2m = np.min(x2)
    x2M = np.max(x2)

    X1 = np.arange(int(x1m),int(x1M),max(1, int(x1M-x1m)/30))
    X2 = np.arange(int(x2m),int(x2M),max(1, int(x2M-x2m)/30))
    X1, X2 = np.meshgrid(X1, X2)

    Y = func(w,X1,X2)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x1, x2, t)
    ax.plot_wireframe(X1, X2, Y)
    plt.show()



data = read_file()
columns = data.columns

w = analysis(data, columns)
x1 = np.array(data[columns[i0]])
x2 = np.array(data[columns[i1]])
t = np.array(data[columns[d]])
plot(w, x1, x2, t)
