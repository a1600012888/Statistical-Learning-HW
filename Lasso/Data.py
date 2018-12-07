import numpy as np
import os
from collections import namedtuple

Data = namedtuple(typename='Data', field_names=['X', 'Y', 'beta'])

def load_data(dir_p = './data'):
    X = np.loadtxt(os.path.join(dir_p, 'X.txt'))
    Y = np.loadtxt(os.path.join(dir_p, 'Y.txt'))
    beta = np.loadtxt(os.path.join(dir_p, 'beta.txt'))

    Y = Y[:, np.newaxis]

    print('Shape of X:{}, Y:{}, beta:{}'.format(X.shape, Y.shape, beta.shape))

    data = Data(X, Y, beta)
    return data

class DataSet(object):
    N = 100
    p = 5000

    def __init__(self, rho=0.3):
        self.rho = rho
        self.Cov = np.eye(self.p)
        for i in range(1, self.p):
            print('now:', i)
            self.Cov += np.eye(self.p, k = i) * self.rho
            self.Cov += np.eye(self.p, k = -i) * self.rho
        self.Mean = np.zeros_like(range(self.p))

        print(self.Mean.shape, self.Cov.shape)
        self.X = np.random.multivariate_normal(self.Mean, self.Cov, self.N)

        beta1 = np.random.uniform(low = 0.5, high=1.0, size=5)
        beta2 = np.random.uniform(low = -1.0, high=-0.5, size = 5)
        beta3 = np.zeros_like(range(len(beta1) + len(beta2), self.p))

        self.beta = np.concatenate((beta1, beta2, beta3), axis = 0)


    def generate(self, sigma = 0.1):
        noise = np.random.normal(loc = 0, scale=sigma, size = self.N)

        self.Y = np.matmul(self.X, self.beta) + noise

        return self.Y

    def save(self, dir_p = './data'):
        np.savetxt(fname=os.path.join(dir_p, 'X.txt'), X = self.X)
        np.savetxt(fname=os.path.join(dir_p, 'Y.txt'), X = self.Y)
        np.savetxt(fname=os.path.join(dir_p, 'beta.txt'),X = self.beta)
        np.savetxt(fname=os.path.join(dir_p, 'Cov.txt'), X = self.Cov)



if __name__ == '__main__':
    a = DataSet()

    print(a.Cov.shape, a.Mean.shape, a.beta.shape)

    Y = a.generate()
    print(Y.shape)

    a.save()



