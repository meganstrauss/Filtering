import numpy as np
import matplotlib as plt
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import random

def vector2matrix(x):
    #takes in vector and returns hermitian matrix
    n = np.shape(x)[0]
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            res[i, j] = x[abs(i-j)]
    return res   
class WienerFilter:
    def __init__(self, z, size, rs, rv):
        self.z = z
        self.rv = rv[0:size] #autocorrelation of noise/noise variance
        self.rsz = rs[0:size] # cross correlation of signal and input
        #self.rz = self.rv+self.rsz #autocorrelation of signal and noise
        #self.rz = self.rz / np.max(self.rz)
        self.rz = np.correlate(self.z, self.z, 'same')[0:size]
        self.rz = self.rz /np.max(self.rz)
        self.size = size
        self.h = self.main()
    def main(self):
        rz = vector2matrix(self.rz) #autocorrelation matrix
        h = np.matmul(np.linalg.inv(rz), self.rsz)
        print(h)
        return h  
    def step(self, i):
        #one step ahead predictor for 
        res = 0
        i -=1
        if i-self.size+1>=0:
            res = np.dot(self.h, self.z[i-self.size+1:i+1])
        return res
def test():
    df = pd.read_csv('austin_weather.csv')
    saved_column = df['TempAvgF'] #you can also use df['column_name']
    saved_column = saved_column.to_numpy()
    #rs = np.correlate(saved_column, saved_column, 'same')
    rs = np.array([1 for i in range(100)])
    rv = np.array([.8, .5, .7, .5, .6, .7, .8, .5, .7, .5, .6, .7])
    w = WienerFilter(saved_column, 7, rs, rv)
    res = []
    for i in range(len(saved_column)):
        res.append(w.step(i))
    t = np.array([i for i in range(len(saved_column))])
    plt.plot(t,saved_column, t, res)
    plt.show()
test()