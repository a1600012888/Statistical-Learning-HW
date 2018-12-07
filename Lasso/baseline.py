from sklearn.linear_model import Lasso

from Data import load_data
import numpy as np

import time
import argparse
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--lamabda', type=float, default=1.0)
    parser.add_argument('--save_path', type=str, default='./Baseline/example.txt')

    args = parser.parse_args()

    save_path = args.save_path
    data = load_data()

    start_time = time.time()
    clf = Lasso(args.lamabda , fit_intercept=True)

    end_time = time.time()

    print('Taking {:.6f} seconds'.format(end_time - start_time))
    beta = clf.fit(data.X, data.Y).coef_

    print("L1: {:.2f}".format(np.linalg.norm(beta, ord=1)))

    np.savetxt(args.save_path, beta)
