#!/usr/bin/python
# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
import math
import random

#データを読み込む
f = open('iris_dataset2.txt')
data1 = f.read()
f.close()
lines1 = data1.split()
data = []
for i in lines1:
    data.append(i.split(','))

#データを数値に変換
n = len(data)
for i in range(n):
    for j in range(5):
        if (j != 4):
            data[i][j] = float(data[i][j])


#kが2から5まで
for k in range(2, 6):
    # c:代表点, c_old:一つ前の代表点
    # pat: アヤメiがどの代表点に最も近いか, dist:代表点iとアヤメjの距離
    # c_list[i]: 代表点iが一番近いアヤメの集合 
    c = [[0 for i in range(4)] for i in range(k)]
    c_old = [[0 for i in range(4)] for i in range(k)]

    while(True):
        for i in range(k):
            ran = random.randint(0,n - 1)
            for j in range(4):
                c[i][j] = data[ran][j]

        flag = True
        for i in range(k):
            for j in range(i+1,k):
                if(c[i] == c[j]):
                    flag=False
        if flag:
            break
    
                
    c_list2 = []
    while(c_old != c):
        pat = []
        dist = []
        c_list = [[] for i in range(k)]
        for i in range(k):
            dist.append([])
            for j in range(n):
                d = 0
                for l in range(4):
                    d += (data[j][l] - c[i][l])**2            
                dist[i].append(d)
                
        for i in range(n):
            M = 1000000000
            M_j = -1
            for j in range(k):
                if M > dist[j][i]:
                    M = dist[j][i]
                    M_j = j
            pat.append(M_j)
            c_list[M_j].append(data[i])
        
        for i in range(k):
            for j in range(4):
                c_old[i][j] = c[i][j]
        
        for i in range(k):
            for j in range(4):
                c[i][j] = 0
        for i in range(n):
            for j in range(4):
                c[pat[i]][j] += data[i][j]
        for i in range(k):
            for j in range(4):
                if(len(c_list[i]) != 0):
                    c[i][j] /= len(c_list[i])
                    
        c_list2 = c_list

    color = ['red', 'blue', 'green', 'yellow', 'black']

    for i in range(k):
        x = []
        y = []
        for j in range(len(c_list2[i])):
            x.append(c_list2[i][j][0])
            y.append(c_list2[i][j][3])
        left = np.array(x)
        height = np.array(y)
        plt.subplot(2, 2, k - 1)
        plt.scatter(left,height,c = color[i])
        plt.scatter(c[i][0], c[i][3], c = color[i], marker = '*', s = 50);
plt.show()
