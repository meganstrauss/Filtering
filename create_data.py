import numpy as np
import random


# f(x, y, z) = x(t-1) + 1, y(t-1) -1 + (1/2)x(t-1), ;



def main():
    noisy = [[10, 0]]
    true = [[10, 0]]
    last = [10, 0]

    for i in range(100):

        x = last[0] +1 
        x1 = x+  np.random.normal(0, 1)
        y = last[1] - 1 + .5*last[0] 
        y1 = y + np.random.normal(0, 1)
        z = last[1]
        z1 = z +  np.random.normal(0, 1)
        noisy.append([np.floor(x1), np.floor(y1), np.floor(z1)])
        true.append([np.floor(x), np.floor(y), np.floor(z)])
        last = [x1, y1, z1]
    print(noisy)
    print("\n")
    print(true)
main()

