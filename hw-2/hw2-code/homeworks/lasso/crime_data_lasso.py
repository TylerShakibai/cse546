if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")
    x_train = df_train.drop('ViolentCrimesPerPop', axis = 1)
    y_train = df_train['ViolentCrimesPerPop']
    x = np.asarray(df_test.drop('ViolentCrimesPerPop', axis = 1))
    y = np.asarray(df_test['ViolentCrimesPerPop'])
    x_test = np.asarray(df_test.drop('ViolentCrimesPerPop', axis = 1))
    y_test = np.asarray(df_test['ViolentCrimesPerPop'])

    lambdaMax = np.max(2 * np.sum(x.T * (y - np.mean(y)), axis=0))
    w = np.zeros((x.shape[1],))
    lambdas = [lambdaMax/(2**i) for i in range(16)]

    noNumZeros, train_mse, test_mse, agePct12t29_ls, pctWSocSec_ls, pctUrban_ls, agePct65up_ls, householdsize_ls = [], [], [], [], [], [], [], []
    for lam in lambdas:
        w = train(x, y, lam, 0.0001, None)[0]
        b = train(x, y, lam, 0.0001, None)[1]
        notZeros = np.count_nonzero(w)
        noNumZeros.append(notZeros)

        y_preds = x.dot(w) + b
        y_test_preds = x_test.dot(w) + b
        train_mse_val = np.mean((y - y_preds)**2 - b)
        test_mse_val = np.mean((y_test - y_test_preds)**2)
        train_mse.append(train_mse_val)
        test_mse.append(test_mse_val)
        
        agePct12t29_ls.append(w[0])
        pctWSocSec_ls.append(w[1])
        pctUrban_ls.append(w[2])
        agePct65up_ls.append(w[3])
        householdsize_ls.append(w[4])

    plt.figure(figsize = (10, 5))
    plt.plot(lambdas, noNumZeros)
    plt.xscale('log')
    plt.xlabel("$\lambda$")
    plt.ylabel("Number of nonzero weights of w")
    plt.show()

    plt.figure(figsize = (10, 5))
    plt.plot(lambdas, agePct12t29_ls, label='agePct12t29')
    plt.plot(lambdas, pctWSocSec_ls, label='pctWSocSec_ls')
    plt.plot(lambdas, pctUrban_ls, label='pctUrban_ls')
    plt.plot(lambdas, agePct65up_ls, label='agePct65up_ls')
    plt.plot(lambdas, householdsize_ls, label='householdsize_ls')
    plt.xscale('log')
    plt.xlabel("$\lambda$")
    plt.ylabel('Coefficient')
    plt.legend()
    plt.show()

    plt.figure(figsize = (10, 5))
    plt.plot(lambdas, train_mse, label = 'Train')
    plt.plot(lambdas, test_mse, label = 'Test')
    plt.xscale('log')
    plt.xlabel("$\lambda$")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
    
    w = train(x, y, 30, 1e-4, None)[0]
    print(x_train.columns[np.argmax(w)])
    print(np.max(w))
    print(x_train.columns[np.argmin(w)])
    print(np.min(w))


if __name__ == "__main__":
    main()