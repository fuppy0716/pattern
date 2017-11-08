#!/usr/bin/python
# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
import math

#データを読み込む
f = open('iris_dataset.txt')
data1 = f.read()
f.close()
lines1 = data1.split()
for i in lines1:
        print i

#読み込んだデータを二次元配列に格納する
data = []
sum = [0.0, 0.0, 0.0, 0.0]
for i in range(len(lines1)):
        if(i%5==0):
                data.append([])
        j = lines1[i]
        if(i%5 != 4):
                data[i/5].append(float(j))
                sum[i%5] += data[i/5][i%5]
        else:
                data[i/5].append(j)

#それぞれの特徴量の和が等しくなるように係数をかける
'''
for i in range(4):
        for j in range(len(data)):
                data[j][i] = data[j][i] * sum[0] / sum[i]
'''
#アヤメiとアヤメjの距離を計算する。距離が小さい順に並べ替える
dis = []
for i in range(len(data)):
        dis.append([])
        for j in range(len(data)):
                if(i==j):
                        dis[i].append((1000000000, j))
                else:
                        d = math.sqrt((data[i][0]-data[j][0])**2 + (data[i][1]-data[j][1])**2 + (data[i][2]-data[j][2])**2 + (data[i][3]-data[j][3])**2)
                        dis[i].append((d, j))

                        
for i in dis:
        i.sort()

        
#kごとの識別率を計算する
rate = []        
for k in range(1,31):
        rate.append(0)
        for i in range(len(dis)):
                cnt0 = 0
                cnt1 = 0
                cnt2 = 0
                for j in range(k):
                      if(data[dis[i][j][1]][4] == "I. setosa"):
                              cnt0 += 1
                      elif(data[dis[i][j][1]][4] == "I. versicolor"):
                              cnt1 += 1
                      else:
                              cnt2 += 1
                              
                if(data[i][4] == "I. setosa"):
                        if(cnt0 > cnt1 and cnt0 > cnt2):
                                rate[k-1] += 1
                elif(data[i][4] == "I. versicolor"):
                        if(cnt1 > cnt0 and cnt1 > cnt2):
                                rate[k-1] += 1
                else:
                        if(cnt2 > cnt0 and cnt2 > cnt1):
                                rate[k-1] += 1
                          
        rate[k-1] = float(rate[k-1])/len(dis)

#kがいくつの時識別率が最大化を求める
M = -1
for k in range(len(rate)):
        M = max(M,rate[k])
for k in range(len(rate)):
        if(M == rate[k]):
                print("{0} {1}".format(k+1,M))
#グラフを書く
left = np.array(range(1,31))
height = np.array(rate)
plt.plot(left,height)
plt.show()
