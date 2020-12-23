import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2


def abs(x):
    if x>=0:
        return x
    else:
        return -x
def mag(x):
    return np.sqrt(x.dot(x))
'''
Weiner fiter
Inputs
    Y = input signal (numpy array(1d??))
    fs = sampling rate(unnecessary??)
    noise = psd of the noise||mean squared noise of signal
Outputs
    Filtered signal Y' with reduced noise based on value of noise given


    can get the noise instead from finding initial silence//noise and then 
    taking average--only works if you have silent segments

    Do i need the covariance and mean of snr?
'''
class WeinerFilter:
    # might need the sampling frequency, dont think i do, possibly for calculating
    #snr or fft.
    def __init__(self, Y, noise, h):
        self.Y = Y
        self.fft = np.fft.fft2(self.Y)
        self.h = h
        self.noise = noise
        self.filter = self.main()

    def main(self):
        h = self.h
        #h /=np.sum(self.h)
        h = np.fft.fft2(h, s = np.shape(self.fft))
        h = (self.fft-self.noise)/h
        v = np.conjugate(h)
        v /= (np.dot(h, v))+10#self.snr
        #v = np.fft.ifft2(v)
        return v # same size as y
def test():
    name = 'mona lisa.jpg'
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (11, 11), 10)
    noise = np.random.normal(0,100,np.shape(image))
    noisy = blurred+noise
    h = np.random.normal(5, 5/3, (5, 1))
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    filter = WeinerFilter(noisy, 11, h)
    new = filter.main()
    new = np.absolute(new)
    print(np.min(new))
    new *= image
    new *=900000
    print(new)
    #new /=np.max(new)
    ret = np.concatenate((image/255, noisy/255), axis=1)
    ret = np.concatenate((ret, new), axis=1)
    cv2.imshow('show', ret)
    cv2.waitKey(0)
test()



          
            

    



