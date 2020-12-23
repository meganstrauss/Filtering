import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import random
from create_data_DKF import *


def matmul(x, y):
    if np.size(x)==1 and np.size(y)==1:
        return np.array([x*y])
    return np.matmul(x, y)

class DKF:
    '''
        d is length of state
        inputs: f: observation space->dx1 mean
                Q: observation space->dxd covariance
                data: observation values
                A, gamma: dxd matrix, dxd matrix st p(zt|zt−1) = ηd(zt;Azt−1,gamma)
        outputs: posterior distribution p(zt|x1:t−1) ≈ ηd(zt;νt,Mt),
    '''
    def __init__(self, f, Q, data, A, gamma, true):
        self.f = f
        self.Q = Q
        self.A = A
        self.gamma = gamma
        self.data = data
        self.states = []
        self.true = true
    def main(self):
        self.state = self.f(self.data[0])
        self.sigma = self.Q(self.data[0])
        for i in range(1, len(self.data)):
            v_t = matmul(self.A, self.state) # Amu_t-1
            M = (matmul(matmul(self.A, self.sigma), np.transpose(self.A)) 
                    + self.gamma) # ASigma_t-1A^t + gamma
            Minv = np.linalg.inv(M) # M^-1
            #Qxinv = np.linalg.inv(np.matmul(self.Q, self.data[i])) # Q(x_t)^-1
            Qxinv = np.linalg.inv(self.Q(self.data[i]))
            sigma_t = np.linalg.inv(Minv + Qxinv) # (M^-1 + Q(x_t)^-1)^-1
            MinvV = matmul(Minv, v_t) # M^-1v_t
            qf = matmul(Qxinv, self.f(self.data[i])) 
                                                             # Q(x_t)^-1f(x_t)
            mu_t = matmul(sigma_t, MinvV + qf) 
                                            # Sigma(M^-1v_t + Q(x_t)^-1f(x_t))
            print(mu_t)
            self.state = mu_t
            self.sigma = sigma_t

            self.states.append(mu_t)
    def show(self):
        t = [i for i in range(101)]
        self.states = np.array(self.states)
        print(self.states)
        print(self.data)
        print(np.size(self.data[:, 0]))
        print(np.size([x[0] for x in self.states]))
        print(self.true)
        plt.plot(t[0:100], [x[0] for x in self.states])
        #plt.plot(t, self.data[:, 0], t, [x[0] for x in self.true], t[0:100], [x[0] for x in self.states])
        # plt.plot(t, [x[0] for x in self.states]) 
        plt.show()
def test():
    result, classes = create()
    gkr = GKR(result, classes, 10)
    A = [1]
    error = [classes[x] - gkr.predict(result[x]) for x in range(len(result))]
    def Q(x):
        return np.array(x)
    P = np.array([[.0001, 0],
                  [0, .0001]])
    def f(x):
        x = gkr.predict(x)
        return np.array([x])
    filter = DKF(f, Q, result, A, P, classes)
    filter.main()
    filter.show()    
test()



            
