from filter_main import *
import cv2
import random

def makeNoisy(image):
    row,col= image.shape
    mean = 0
    var = 100
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return (noisy, gauss)

def main():
    desired = cv2.imread('Flag-Japan.jpg')
    desired = cv2.cvtColor(desired, cv2.COLOR_BGR2GRAY)
    G = np.zeros_like(desired)
    G = G.astype(complex)
    l = 10
    '''
    for i in range(l):
        new, noise = makeNoisy(desired)
        f = filter(new, noise, desired)
        newg = f.filter
        G+= newg
    G/=l

    '''
    noisy, noise = makeNoisy(desired)
    filt = filter(noisy, noise, desired, h)
    G = filt.filter
    new = np.matmul(G, noisy)
    new = np.concatenate((new,desired), axis=1)
    new = np.concatenate((new, noisy), axis=1)
    new = np.absolute(new)
    new /= 255
    cv2.imshow('new', new)
    cv2.waitKey(0)
def main2():
    image = cv2.imread('mona lisa.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (11, 11), 10)
    noisy, noise = makeNoisy(blurred)
    filt = WeinerFilter(noisy, noise, image)
    f = filt.filter
    new = f*noisy
    new = np.abs(new)
    #new /= np.max(new)
    #new /=255
    new = np.absolute(new)
    ret = np.concatenate((image/255, noisy/255), axis=1)
    ret = np.concatenate((ret, new), axis=1)
    #ret /=255
    cv2.imshow('show', ret)
    cv2.waitKey(0)
main2()
    
    
        
