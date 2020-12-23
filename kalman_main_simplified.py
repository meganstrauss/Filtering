import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import random


class Kalman:
    def __init__(self, z, x0, f, v, sw):
        # f is next estimated state from estimated state function
        # v is noise variance
        self.x = x0
        self.z = z
        self.f = f # State transition matrix
        self.v = 1/2
        self.v0 = 50 #this is wrong
        self.sw = sw # process noise variance
        self.filtered = np.zeros_like(self.z)
        self.i = 0 # counter for which measurement value to check next
    def filter(self):
        while self.i<len(self.z):
            #update
            new = self.z[self.i]
            k = self.v/(self.v+self.v0)
            x1 = self.x+(k*(new-self.x))
            v1 = self.v*(1-k)
            self.filtered[self.i] = x1
            #predict
            self.x = self.f*x1
            self.v = v1 + self.sw
            self.i+= 1
        return self.filtered
    def show(self):
        t = np.array([i for i in range(len(self.z))])
        plt.plot(t,self.z, t, self.filtered)
        plt.show()
def test():
    df = pd.read_csv('austin_weather.csv')
    saved_column = df['TempAvgF'] #you can also use df['column_name']
    saved_column = saved_column.to_numpy()
    rs = np.correlate(saved_column, saved_column, 'same')
    rv = np.array([random.uniform(.5, .8) for i in range(100)])
    k = Kalman(saved_column, saved_column[0], 1, .88, .2)
    k.filter()
    k.show()

test()
        
