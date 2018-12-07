import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
from Data import load_data
def read_beta(dir_path = './SGDbeta-V3'):
    lamabdas = list(range(100))

    B = []
    L1 = []
    for la in lamabdas:
        path = os.path.join(dir_path, 'lambda.{}.txt'.format(la))
        beta = np.loadtxt(path)

        l1 = np.linalg.norm(beta, ord = 1)

        B.append(beta)
        L1.append(l1)

    B = np.array(B)
    L1 = np.array(L1)

    return B, L1


def ShowLambda(B, alpha = 0.85):

    assert 0 <= alpha and alpha <= 1, 'Alpha out of range'
    lamabdas = np.linspace(start=-5, stop=7, num=100)
    lamabdas = np.exp(lamabdas)

    stop = int(alpha * 100)
    plt.figure()

    plt.plot(lamabdas[:stop], B[:stop,])

    plt.xlabel('Lambda')

    plt.show()




def ShowL1(B, L1):

    plt.plot(L1, B)

    plt.xlabel('L1 norm of beta')

    plt.show()


def ShowReconstruction(B, L1, data):
    pred = data.X @ B.transpose()
    print(pred.shape)

    residual = pred - data.Y

    loss = np.linalg.norm(residual, axis=0, ord=2)

    loss = loss * loss / 0.5

    plt.figure()

    plt.plot(L1, loss, label = 'Residual term')

    plt.legend(loc = 'upper right')

    plt.xlabel('L1 norm of beta')
    plt.ylabel('Reconstruction Loss')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type = str, default='./SGDBeta-V3')
    args = parser.parse_args()

    B, L1 = read_beta(args.dir_path)

    data = load_data()
    ShowLambda(B, alpha=0.85)
    ShowL1(B, L1)
