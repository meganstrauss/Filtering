import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import random
from filter_main3 import *


''' inputs:
        z: nxm matrix
        x0: v length vector (starting state, v is number of state variables)
        v0: v length vector (starting variance)
        F: function that inputs state and outputs state (vxv matrix ?)
        H: takes in state outputs observation? (vxv matrix?)
        Q: nxn matrix
        R: same size as H
    DO OBSERVATION AND STATE HAVE TO BE SAME SIZE
'''
class Kalman:
    def __init__(self, z, x0, v0, F, H, Q, R):
        # f is next estimated state from estimated state function
        # v is noise variance
        self.xp = x0 # initial state
        self.v = v0# same size as h doesnt really matter what starting value is
        self.z = z # observation set
        self.f = F # State transition model
        self.h = H # observation model
        self.q = Q # covariance of process noise
        self.r = R # covairance of observation noise
        self.filtered = np.zeros_like(self.z)
        self.i = 0 # counter
    def filter(self):
        while self.i<len(self.z):
            # update
            y = self.z[i]-np.matmul(self.h, self.xp) #estimate & observed value difference
            k = np.matmul(np.matmul(self.v, np.transpose(self.h)), 
                np.linalg.inv(self.r + np.matmul(np.matmul(self.h, self.v), 
                np.transpose(self.h)))) # vH^T(R+HvH^T)^-1
                
            x = self.xp+matmul(k, y) #final signal estimate
            hk = np.matmul(k, self.h)
            identity = np.identity(np.shape(h1)[0])
            v = np.matmul((identity-hk), self.v) #final noise estimate
            # prediction
            self.xp = np.matmul(self.f, x) #prediction of next state
            self.v = np.matmul(np.matmul(self.f, v), np.transpose(self.f)) + self.q
        return self.filtered
def test():

    saved_column = [(c1[i],c2[i]) for i in range(len(c1))]
    f = np.array([[1,0],[0,1]])
    # want f to be identity
    h = np.array([[1,0],[0,1]])
    # want h to be some transformation
    rs = np.correlate(saved_column, saved_column, 'same')
    rv = np.array([random.uniform(.5, .8) for i in range(100)])
    k = Kalman(saved_column, saved_column[0], , f, f, 3)
    k.filter()
    k.show()

test()
        
