#!/usr/bin/env python3

import sklearn.datasets as datasets
import numpy as np
from matplotlib import pyplot as plt


def predict(X, w):
    # TODO: Your code here
    y_hat = np.dot(X, w)
    return y_hat


# TODO needs normalized inputs
def calculate_loss(normalized_y_hat, normalized_y):
    M = normalized_y_hat.shape[0]
    loss = (1 / (2 * M)) * np.sum(np.power((normalized_y_hat - normalized_y), 2))
    return loss


def calculate_risk(original_y_hat, original_y):
    M = original_y.shape[0]
    risk = (1 / M) * np.sum(abs(original_y_hat - original_y))
    return risk


def train(X_train, y_train, X_val, y_val, MaxIter, batch_size, alpha):
    N_train = X_train.shape[0]
    # N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0

    for epoch in range(MaxIter):
        loss_this_epoch = 0
        num_batches = int(np.ceil(N_train / batch_size))
        for b in range(num_batches):

            X_batch = X_train[b * batch_size : (b + 1) * batch_size]
            y_batch = y_train[b * batch_size : (b + 1) * batch_size]

            y_hat_batch = predict(X_batch, w)
            loss_batch = calculate_loss(y_hat_batch, y_batch)
            loss_this_epoch += loss_batch

            # TODO: Your code here
            # Mini-batch gradient descent
            gradient = (X_batch.T.dot(y_hat_batch - y_batch)) / batch_size
            w = w - alpha * gradient

        # 1. Compute the training loss by averaging loss_this_epoch
        avg_training_loss = loss_this_epoch / num_batches

        # 2. Perform validation on the validation test by the risk
        y_hat_validation = predict(X_val, w)
        validation_risk = calculate_risk(y_hat_validation, y_val)

        # 3. Keep track of the best validation epoch, risk, and the weights
        risks_val.append(validation_risk)
        losses_train.append(avg_training_loss)
        if validation_risk < risk_best:
            risk_best = validation_risk
            epoch_best = epoch
            w_best = w

    # Return some variables as needed
    return epoch_best, w_best, risk_best, risks_val, losses_train


def test(X_test, y_test, w):
    y_hat_test = predict(X_test, w)
    risk = calculate_risk(y_hat_test, y_test)
    return risk


############################
# Main code starts here
############################
def main():
    # Load data. This is the only allowed API call from sklearn
    X, y = datasets.load_boston(return_X_y=True)
    y = y.reshape([-1, 1])
    # X: sample x dimension
    # y: sample x 1

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Augment feature
    X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
    # X_: Nsample x (d+1)

    # normalize features:
    mean_y = np.mean(y)
    std_y = np.std(y)
    y = (y - mean_y) / std_y

    # print(X.shape, y.shape) # It's always helpful to print the shape of a variable

    # Randomly shuffle the data
    np.random.seed(314)
    np.random.shuffle(X_)
    np.random.seed(314)
    np.random.shuffle(y)

    X_train = X_[:300]
    y_train = y[:300]

    X_val = X_[300:400]
    y_val = y[300:400]

    X_test = X_[400:]
    y_test = y[400:]

    #####################
    # setting

    alpha = 0.001  # learning rate
    batch_size = 10  # batch size
    MaxIter = 100  # Maximum iteration

    # TODO: Your code here
    epoch_best, w_best, risk_best, risks_val, losses_train = train(
        X_train, y_train, X_val, y_val, MaxIter, batch_size, alpha
    )
    print("best epoch: ", epoch_best, "best w: ", w_best, "best risk: ", risk_best)
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("validation risk")
    plt.plot(risks_val, color="red")
    plt.savefig("risks_over_epoches")
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.plot(losses_train, color="blue")
    plt.savefig("traning_loss_over_epoches")

    test_risk = test(X_test, y_test, w_best)
    print("test risk: ", test_risk)

    # Perform test by the weights yielding the best validation performance

    # Report numbers and draw plots as required.


main()
