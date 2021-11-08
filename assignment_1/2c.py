#!/usr/bin/env python3

import sklearn.datasets as datasets
import numpy as np
from matplotlib import pyplot as plt


def predict(X, w):
    # TODO: Your code here
    y_hat = np.dot(X, w)
    return y_hat


# needs normalized inputs
def calculate_loss(normalized_y_hat, normalized_y):
    M = normalized_y_hat.shape[0]
    loss = (1 / (2 * M)) * np.sum(np.power((normalized_y_hat - normalized_y), 2))
    return loss


# data should be denormalized for this function
def calculate_risk(original_y_hat, original_y):
    M = original_y.shape[0]
    risk = (1 / M) * np.sum(abs(original_y_hat - original_y))
    return risk


def denormalize_data(normalized_data, data_mean, data_std):
    temp = normalized_data * data_std
    return temp + data_mean


def train(training_info, X_val, y_val, MaxIter, batch_size, alpha):
    X_train = training_info["X_train"]
    y_train = training_info["y_train"]

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
        # denormalize y_hat and y_test to calculate risks
        mean_y = training_info["mean_y"]
        std_y = training_info["std_y"]
        denormalized_y_hat_validation = denormalize_data(
            y_hat_validation, mean_y, std_y
        )
        denormalized_y_val = denormalize_data(y_val, mean_y, std_y)
        validation_risk = calculate_risk(
            denormalized_y_hat_validation, denormalized_y_val
        )

        # 3. Keep track of the best validation epoch, risk, and the weights
        risks_val.append(validation_risk)
        losses_train.append(avg_training_loss)
        if validation_risk < risk_best:
            risk_best = validation_risk
            epoch_best = epoch
            w_best = w

    # Return some variables as needed
    return epoch_best, w_best, risk_best, risks_val, losses_train


def test(X_test, y_test, w, y_mean, y_std):
    y_hat_test = predict(X_test, w)
    denormalized_y_test = denormalize_data(y_test, y_mean, y_std)
    denormalized_y_hat_test = denormalize_data(y_hat_test, y_mean, y_std)
    risk = calculate_risk(denormalized_y_hat_test, denormalized_y_test)
    loss = calculate_loss(y_hat_test, y_test)
    print("test loss: ", loss)
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
    mean_x = np.mean(X, axis=0)
    std_x = np.std(X, axis=0)
    X = (X - mean_x) / std_x

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
    test_risks = []
    learning_rates = []
    for alpha in np.arange(0.00001, 0.1, 0.001):  # learning rate
        print("learning rate: ", alpha)
        batch_size = 10  # batch size
        MaxIter = 100  # Maximum iteration

        # TODO: Your code here
        traning_info = {
            "X_train": X_train,
            "y_train": y_train,
            "mean_x": mean_x,
            "std_x": std_x,
            "mean_y": mean_y,
            "std_y": std_y,
        }
        epoch_best, w_best, risk_best, risks_val, losses_train = train(
            traning_info, X_val, y_val, MaxIter, batch_size, alpha
        )
        # print("best epoch: ", epoch_best, "best w: ", w_best, "best risk: ", risk_best)

        test_risk = test(X_test, y_test, w_best, mean_y, std_y)
        test_risks.append(risk_best)
        learning_rates.append(alpha)
        print("test risk: ", test_risk)

        # Perform test by the weights yielding the best validation performance

        # Report numbers and draw plots as required.
    print(test_risks)
    plt.figure()
    plt.xlabel("learning_rate")
    plt.ylabel("validation_best_risk")
    plt.plot(learning_rates, test_risks, color="blue")
    plt.savefig("_rates")
    print("best: ", min(test_risks), learning_rates[test_risks.index(min(test_risks))])


main()
