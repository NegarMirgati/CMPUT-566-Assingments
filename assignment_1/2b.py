#!/usr/bin/env python3

import sklearn.datasets as datasets
import numpy as np
from matplotlib import pyplot as plt


def predict(X, w):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

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


def train(X_train, y_train, X_val, y_val):
    N_train = X_train.shape[0]
    # N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    best_decay_risk = 10000
    best_decay_epoch = 0
    best_decay_w = 0
    best_decay = -1
    best_decay_losses = []
    best_decay_risks_val = []
    for decay in decay_list:
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
                w = w - alpha * (X_batch.T.dot((y_hat_batch - y_batch)) + decay * w)

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

        if risk_best < best_decay_risk:
            best_decay_epoch = epoch_best
            best_decay_w = w_best
            best_decay_risk = risk_best
            best_decay_risks_val = risks_val
            best_decay = decay
            best_decay_losses = losses_train

    # Return some variables as needed
    return (
        best_decay_epoch,
        best_decay_w,
        best_decay_risk,
        best_decay_risks_val,
        best_decay,
        best_decay_losses,
    )


def test(X_test, y_test, w):
    y_hat = predict(X_test, w)
    risk = calculate_risk(y_hat, y_test)
    return risk


############################
# Main code starts here
############################

# Load data. This is the only allowed API call from sklearn
X, y = datasets.load_boston(return_X_y=True)
y = y.reshape([-1, 1])
# X: sample x dimension
# y: sample x 1
print(X.shape)
X_powered = np.power(X, 2)
X = np.hstack((X, X_powered))
print(X.shape)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Augment feature
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)

y = (y - np.mean(y)) / np.std(y)

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
decay_list = {3, 1, 0.3, 0.1, 0.03, 0.01}
decay = 0.0  # weight decay


# TODO: Your code here
(
    best_decay_epoch,
    best_decay_w,
    best_decay_risk,
    best_decay_risks_val,
    best_decay,
    best_decay_losses,
) = train(X_train, y_train, X_val, y_val)
print(
    "best epoch: ",
    best_decay_epoch,
    "best w: ",
    best_decay_w,
    "best risk: ",
    best_decay_risk,
    "best decay: ",
    best_decay,
)
plt.figure()
plt.xlabel("epoch")
plt.ylabel("validation risk")
plt.plot(best_decay_risks_val, color="red")
plt.savefig("hyperparameterized_risks")
plt.figure()
plt.xlabel("epoch")
plt.ylabel("Training loss")
plt.plot(best_decay_losses, color="blue")
plt.savefig("hyperparameterized_losses")

# test_output = test(X_test, y_test, w_best)
# print("teeeest", test_output)


# Perform test by the weights yielding the best validation performance

# Report numbers and draw plots as required.
