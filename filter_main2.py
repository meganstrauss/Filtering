import numpy as np
import matplotlib as plt
import cv2

class WeinerFilter:
    def __init__(self, y, h, isnr):
        self.h = h
        self.y = y
        self.fft = np.fft.fft2(y)
        self.isnr = isnr
    def main(self):
        self.h /=np.sum(self.h)
        h = np.fft.fft2(self.h, s = np.shape(self.fft))
        h = np.conjugate(h)/(np.abs(h)**2+self.isnr)
        ret = self.fft*h
        ret = np.fft.ifft2(ret)
        ret = np.abs(ret)
        return ret # same size as y
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
    filter = WeinerFilter(noisy, h, 11)
    new = filter.main()
    #new = np.absolute(new)
    new /=np.max(new)
    ret = np.concatenate((image/255, noisy/255), axis=1)
    ret = np.concatenate((ret, new), axis=1)
    #ret /=255
    cv2.imshow('show', ret)
    cv2.waitKey(0)
test()








