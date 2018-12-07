import numpy as np
from data import get_train_dataset, get_test_dataset
from tqdm import tqdm
import argparse
class GaussianKernel(object):


    def __init__(self, x0:np.ndarray, sigma = 0.5):

        self.sigma = sigma
        self.x0 = x0

    def __call__(self, x):
        '''

        :param x: x can both be a single vector or a list of vectors
        :return:
        '''

        v = x - self.x0
        #v = np.dot(v, v.transpose()) /2.0
        v = np.linalg.norm(v, ord = 2, axis = -1) ** 2
        v = v / (-2.0 * self.sigma)

        v = np.exp(v)
        return v



def GetStatistics(X, Y, Kernel):
    Mean = []
    Pi = []
    Cov = np.zeros((X.shape[1], X.shape[1]))
    for i in range(10):
        idx = (Y == i)
        Xi = X[idx]
        kernels = Kernel(Xi)

        PiUnNormalized = np.sum(kernels)


        diag = np.diag(kernels)
        meani = np.matmul(kernels.transpose(), Xi) / PiUnNormalized
        XiMinus_mean = Xi - meani


        Cov = Cov + XiMinus_mean.transpose() @ diag @ XiMinus_mean
        Mean.append(meani)
        Pi.append(PiUnNormalized)
    Mean = np.array(Mean)
    Pi = np.array(Pi)
    Cov = Cov / np.sum(Pi)
    Pi = Pi / np.sum(Pi)

    return Cov, Mean, Pi



def fit(x0, data_train, sigma):

    kernel = GaussianKernel(x0, sigma)

    Cov, Mean, Pi = GetStatistics(data_train.X, data_train.Y, kernel)


    #Rho = np.linalg.inv(Cov + np.eye(256))  This works well

    Rho = np.linalg.pinv(Cov + np.eye(256))



    X = x0 - Mean


    Pred = X @ Rho @ X.transpose()

    preds = np.diag(Pred) *(- 0.5) + np.log(Pi)

    pred = np.argmax(preds, axis = 0)

    return pred


def Eval(X, Y, ds_train, sigma = 1.0):

    now = 0
    right_count = 0

    pbar = tqdm(zip(X, Y))
    for x, y in pbar:
        now = now + 1
        pred = fit(x, ds_train, sigma)
        if pred == y:
            right_count = right_count + 1

        pbar.set_postfix(Accuracy = '{:.3f}'.format(right_count * 1.0 / now))

    Accuracy = right_count * 1.0 / now
    return Accuracy

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type = float, default = 1.0)
    ds_train = get_train_dataset()
    ds_test = get_test_dataset()
    args = parser.parse_args()

    #TrainingAccuracy = Eval(ds_train.X, ds_train.Y, ds_train, args.sigma)
    TestAccuracy = Eval(ds_test.X, ds_test.Y, ds_train, args.sigma)

    #print('Fnishes evaluating!  Accuracy on training set:{:.3f}, Accuracy on test set:{:.3f}'.format(
    #    TrainingAccuracy, TestAccuracy
    #))
    print('Fnishes evaluating!  Accuracy on test set:{:.3f}'.format(
         TestAccuracy
    ))






