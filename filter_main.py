import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


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
        self.noise = np.fft.fft2(noise)
        self.fft = np.fft.fft2(self.Y)
        if len(np.shape(Y))==2:
            self.snr = self.getSNR2d() #inverse of signal to noise ratio
        else:
            self.snr = self.getSNR1d() #inverse of signal to noise ratio
        self.h = h
        self.filter = self.main()

    def main(self):
        h = np.fft.fft2(self.h)
        h = (self.fft-self.noise)/h
        v = np.conjugate(h)
        v /= (np.dot(h, v))+10#self.snr
        #v = np.fft.ifft2(v)
        #a = np.absolute(h)**2
        #y = self.fft #array
        #denom = h*(a+self.snr) #computes elementwise division
        return v # same size as y

    def getSNR1d(self):
        res = np.zeros_like(self.fft)
        noise = np.zeros_like(self.noise)
        for i in range(np.size(self.noise)):
            val = self.noise[i]
            noise[i] = np.abs(val)**2
        for i in range(np.size(self.fft)):
            val = self.fft[i]
            res[i] = np.abs(val)**2
        snr = noise/res
        return snr
    def getSNR2d(self):
        #should compute psd of signal and divide it by the noise value given
        #A PSD is computed by multiplying each frequency bin in an FFT by 
        #its complex conjugate which results in the real only spectrum of 
        #amplitude in g2.
        res = np.zeros_like(self.fft)
        noisefft = self.noise
        noise = np.zeros_like(noisefft)
        for i in range(np.shape(noisefft)[0]):
            for j in range(np.shape(noisefft)[1]):
                val = self.noise[i, j] #i think this notation is wrong
                noise[i, j] = val*np.conj(val)
        for i in range(np.shape(self.fft)[0]):
            for j in range(np.shape(self.fft)[1]):
                val = self.Y[i, j] #i think this notation is wrong
                res[i, j] = val*np.conj(val)
        snr = noise/res
        self.noisepower = noise
        return snr #returns snr for each section
    def show(self):
        #taken from https://matplotlib.org/3.2.2/gallery/lines_bars_and_markers/cohere.html
        fig, axs = plt.subplot(2, 1, 1)
        axs[1].plot(t, self.desired, t, self.filter*self.Y) #need to define t
        axs[1].set_xlim(0, 2)
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('filtered and desired')
        axs[1].grid(True)
        axs[0].plot(t, self.desired, t, self.Y)
        axs[0].set_xlim(0, 2)
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('original signal and desired signal')
        axs[0].grid(True)



          
            

    



