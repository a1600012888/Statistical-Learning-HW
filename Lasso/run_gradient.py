import os
import time
import numpy as np

if __name__ == '__main__':

    lamabdas = np.linspace(start = -5, stop = 7, num = 100)
    lamabdas = np.exp(lamabdas)
    time.sleep(0.5)
    i = 0
    for la in lamabdas:
        print('Now:', la)
        command = 'python3 Gradient.py --device=1 --lamabda={} --save_path={}'.format(
            la, os.path.join('./SGDBeta-V3', 'lambda.{}.txt'.format(i)))
        os.system(command)
        i = i + 1

