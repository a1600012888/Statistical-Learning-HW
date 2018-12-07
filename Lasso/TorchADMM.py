import os
import numpy as np
from Data import load_data, Data
import copy
from tqdm import tqdm
import argparse
import torch
import time


def SoftThreshold(x : torch.Tensor,  sigma):

    x_abs = torch.abs(x) -sigma
    x_abs = torch.where(x_abs <= 0, torch.zeros_like(x_abs), x_abs)
    #x_abs = x_abs * (x_abs > 0)
    y = torch.sign(x) * x_abs

    return y


class ADMM(object):

    def __init__(self, X, Y, lamabda, device, rho = 1):

        self.X = X
        self.lamabda = lamabda
        self.rho = rho
        self.Y = Y
        self.u = np.zeros_like(range(X.shape[1]))[:, np.newaxis]

        self.M = np.linalg.inv((np.matmul(X.transpose(), X) + rho * np.eye(X.shape[1])))
        self.XT = self.X.transpose()

        self.projection = np.matmul(self.XT, Y)

        self.X = torch.tensor(self.X).type(torch.FloatTensor).to(DEVICE)
        self.Y = torch.tensor(self.Y).type(torch.FloatTensor).to(DEVICE)
        self.M = torch.tensor(self.M).type(torch.FloatTensor).to(DEVICE)
        self.projection = torch.tensor(self.projection).type(torch.FloatTensor).to(DEVICE)
        self.u = torch.tensor(self.u).type(torch.FloatTensor).to(DEVICE)
        self.rho = torch.tensor(self.rho).type(torch.FloatTensor).to(DEVICE)
        self.lamabda = torch.tensor(self.lamabda).type(torch.FloatTensor).to(DEVICE)
        self.device = device

        #print(self.X.size(), self.Y.size(), self.projection.size(), self.u.size())
    def step(self, beta, v):

        #start_time = time.time()
        beta = torch.matmul(self.M, self.projection + (v - self.u) * self.rho)
        #beta = np.dot(self.M, self.projection + v - self.u)


        v = SoftThreshold( beta + self.u, self.lamabda / self.rho)

        self.u = self.u + (beta - v) * self.rho

        #end_time = time.time()
        #print('Taking {:.6f} s for on setp'.format(end_time - start_time))
        return beta, v

    def get_residual(self, beta):
        residual = self.Y - torch.matmul(self.X, beta)

        return residual


if __name__ == '__main__':

    TOTAL_ITER = 10000

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=int, default=0,
                        help='Using which gpu?(Should be one of 0-7; Default:0)')
    parser.add_argument('--lamabda', type=float, default=1.0)
    parser.add_argument('--save_path', type=str, default='./ADMMBeta/example.txt')
    parser.add_argument('--continue_beta', type=str, default = './lambda.2000.0.txt')

    args = parser.parse_args()
    DEVICE = torch.device('cuda', args.device)

    save_path = args.save_path
    data = load_data()

    admm = ADMM(data.X, data.Y, args.lamabda, DEVICE)


    if args.continue_beta is not None and os.path.isfile(args.continue_beta):
        beta = np.loadtxt(args.continue_beta)[:, np.newaxis]
        print('Warm Starting -> {}'.format(args.continue_beta))
    else:
        beta = np.random.normal(0, 0.5, data.X.shape[1])

    TOTAL_ITER = int(TOTAL_ITER * (1 + args.lamabda / 50))

    v = np.array(copy.deepcopy(beta))

    beta = torch.tensor(beta).type(torch.FloatTensor).to(DEVICE)
    v = torch.tensor(v).type(torch.FloatTensor).to(DEVICE)

    #print(beta.size(), v.size())

    pbar = tqdm(range(TOTAL_ITER))

    with torch.no_grad():
        for cur_iter in pbar:


            beta, v = admm.step(beta, v)

            distance = torch.norm(beta - v).item()

            residual = admm.get_residual(beta)

            residual_loss = torch.norm(residual).item()
            residual_loss = (residual_loss ** 2) / 2.0


            L1 = torch.norm(beta, p = 1).item()

            pbar.set_postfix(redisual_loss='{:.2f}'.format(residual_loss),
                         l1 = '{:.2f}'.format(L1), dis = "{:.3f}".format(distance))

    beta = beta.detach().cpu().numpy()
    np.savetxt(save_path, beta)
