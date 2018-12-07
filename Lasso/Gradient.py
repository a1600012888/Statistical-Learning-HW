from Data import load_data, Data
import numpy as np
import torch
import os

import torch.nn as nn
import torch.optim as optim
from utils import Affine, L1Penlty, MultiStageLearningRatePolicy, LinearLearningRatePolicy
import argparse
from tqdm import tqdm


def adjust_learning_rate(optimzier, lr):

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_one_iter(data, criterion, lamabda, optimizer, afine):

    optimizer.zero_grad()
    pred = afine(data.X)
    residual = criterion(pred, data.Y) * 0.5
    l1 =  L1Penlty(afine)
    loss = residual + l1 * lamabda

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return residual.item(), l1.item()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type = int, default=0,
                      help='Using which gpu?(Should be one of 0-7; Default:0)')
    parser.add_argument('--lamabda', type = float, default=1.0)
    parser.add_argument('--save_path', type = str, default='./SGDBeta-V2/')
    parser.add_argument('--continue_beta', type = str, default='./lambda.2000.0.txt')
    args = parser.parse_args()

    DEVICE = torch.device('cuda', args.device)
    save_path = args.save_path

    TOTAL_ITER = 200000
    InitiaLearningRate = 1e-5 / (1 + args.lamabda / 20)



    affine = Affine()

    if args.continue_beta is not None and os.path.isfile(args.continue_beta):
        print('Warm starting from {}'.format(args.continue_beta))
        beta = np.loadtxt(args.continue_beta).astype(np.float32)[np.newaxis,:]

        affine.fc.weight = torch.nn.Parameter(torch.tensor(beta))
        TOTAL_ITER = TOTAL_ITER // 10
        InitiaLearningRate = InitiaLearningRate / 2
        GetLearningRate = LinearLearningRatePolicy(TOTAL_ITER, InitiaLearningRate)
    else:
        GetLearningRate = LinearLearningRatePolicy(TOTAL_ITER, InitiaLearningRate)

    affine = affine.to(DEVICE)

    criterion = nn.MSELoss(reduce = True, size_average = False).to(DEVICE)

    optimizer = optim.SGD(affine.parameters(), lr = GetLearningRate(0),momentum = 0.8)

    numpy_data = load_data()

    X = torch.from_numpy(numpy_data.X).type(torch.FloatTensor).to(DEVICE)
    Y = torch.tensor(numpy_data.Y).type(torch.FloatTensor).to(DEVICE)
    beta = torch.tensor(numpy_data.beta).type(torch.FloatTensor).to(DEVICE)
    data = Data(X, Y, beta)
    pbar = tqdm(range(TOTAL_ITER))

    print('Lambda:{}  -- Total Iters:{} -- Initial Lr:{}'.format(args.lamabda, TOTAL_ITER, InitiaLearningRate))
    for cur_iter in pbar:
        residual_loss, l1 = train_one_iter(data, criterion, args.lamabda, optimizer, affine)


        pbar.set_postfix(residual_loss = '{:.2f}'.format(residual_loss),
                         l1 = '{:.2f}'.format(l1))

    beta = affine.fc.weight.detach().cpu().numpy()

    np.savetxt(save_path, beta)

if __name__ == "__main__":
    main()