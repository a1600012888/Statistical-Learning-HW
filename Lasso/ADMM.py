import os
import numpy as np
from Data import load_data, Data
import copy
from tqdm import tqdm
import argparse

import time

def SoftThreshold(x : np.ndarray, sigma = 1.0):
    x_abs = np.abs(x) -sigma
    #x_abs[np.where(x_abs <= 0)] = 0
    x_abs = x_abs * (x_abs > 0)
    y = np.sign(x) * x_abs

    return y


class ADMM(object):

    def __init__(self, X, Y, lamabda):

        self.X = X
        self.lamabda = lamabda
        self.Y = Y
        self.u = np.ones_like(range(X.shape[1]))[:, np.newaxis]

        self.M = np.linalg.inv((np.matmul(X.transpose(), X) + np.eye(X.shape[1])))
        self.XT = self.X.transpose()

        self.projection = np.matmul(self.XT, Y)

    def step(self, beta, v):

        #start_time = time.time()
        beta = np.dot(self.M, self.projection + v - self.u)
        v = SoftThreshold( beta + self.u, self.lamabda)

        self.u = self.u + beta - v

        #end_time = time.time()
        #print('Taking {:.6f} s for one step'.format(end_time - start_time))
        return beta, v

    def get_residual(self, beta):
        residual = self.Y - np.matmul(self.X, beta)

        return residual


if __name__ == '__main__':

    TOTAL_ITER = 10000

    parser = argparse.ArgumentParser()
    parser.add_argument('--lamabda', type=float, default=1.0)
    parser.add_argument('--save_dir', type=str, default='./ADMMBeta/example.txt')
    parser.add_argument('--continue_beta', type=str, default = './lambda.500.0.txt')

    args = parser.parse_args()
    save_path = os.path.join(args.save_dir, 'lambda.{}.txt'.format(args.lamabda))

    data = load_data()

    admm = ADMM(data.X, data.Y, args.lamabda)


    if args.continue_beta is not None and os.path.isfile(args.continue_beta):

        beta = np.loadtxt(args.continue_beta)[:, np.newaxis]
        print('Warm Starting -> {}'.format(args.continue_beta))
    else:
        beta = np.random.normal(0, 0.5, data.X.shape[1])

    TOTAL_ITER = int(TOTAL_ITER * (1 + args.lamabda / 50))
    v = np.array(copy.deepcopy(beta))

    pbar = tqdm(range(TOTAL_ITER))

    for cur_iter in pbar:
        beta, v = admm.step(beta, v)

        residual = admm.get_residual(beta)

        residual_loss = 0.5 * (np.linalg.norm(residual, ord = 2) ** 2)

        L1 = np.linalg.norm(beta, ord = 1)

        pbar.set_postfix(redisual_loss='{:.2f}'.format(residual_loss),
                         l1 = '{:.2f}'.format(L1))
