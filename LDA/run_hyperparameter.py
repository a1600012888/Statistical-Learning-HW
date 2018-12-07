import os
import numpy as np
if __name__ == '__main__':

    sigmas = np.linspace(start = 0.5, stop = 0.9, num=5)
    sigmas2 = np.linspace(start = 2, stop = 10, num=9)
    sigmas3 = np.linspace(start = 15, stop = 50, num=8)
    sigmas = np.concatenate((sigmas, sigmas2, sigmas3), axis = 0)
    print(sigmas)
    for sig in sigmas:
        print(sig)
        command = 'python3 fit.py --sigma={}'.format(sig)
        os.system(command)


