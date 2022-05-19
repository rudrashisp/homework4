import numpy as np

import numpy.matlib

import matplotlib.pyplot as plt

import csv

from statistics import mean

from statistics import stdev

data = np.genfromtxt("new_data.txt", delimiter=' ')

data

X = data[:, 0:2]  # The three columns are in X

Y = data[:, 2]

print(X)

print(Y)

"""Plotting to see if the data is linearly separable"""

for i in range(X.shape[0]):

    if (Y[i] == -1):

        plt.scatter(X[i, 0], X[i, 1], color='red')

    else:

        plt.scatter(X[i, 0], X[i, 1], color='green')

plt.show()

alpha = []

"""Calculating total number of positive and negative instances for Y and setting the alphai for +Yi as 1/no of +instances

   and -Yi=1/no of -ve instances"""

ctr1 = sum([1 if i == 1 else 0 for i in Y])

ctr2 = sum([1 if i == -1 else 0 for i in Y])

"""Setting initial values for alpha"""

for i in range(len(Y)):

    if (Y[i] == 1):
        alpha.insert(i, 1 / ctr1)

    if (Y[i] == -1):
        alpha.insert(i, 1 / ctr2)

alpha = np.array(alpha)

W = np.matmul(alpha * Y, X)  # Calculating weights

b = 0  # bias

t1 = Y * np.dot(X, W) + b - 1  #

KKT = alpha * t1  # Step 3 KKT condition

max = KKT[0]

"""Step 4 a"""

for i in range(len(KKT)):

    if (max < KKT[i]):
        max = KKT[i]

        pos = i

i1 = pos  # Step 3b

E = []

"""Step 3b"""

for i in range(len(X)):
    Kji = np.dot(X, X[i])

    t2 = alpha * Y

    E.insert(i, np.matmul(t2, Kji) - Y[i])

e = []

"""Step 4c"""

for i in range(len(X)):
    Kji = np.dot(X, X[i])

    Kji1 = np.dot(X, X[i1])

    t2 = alpha * Y

    e.insert(i, np.matmul(t2, (Kji1 - Kji)) + Y[i] - Y[i1])

max2 = e[0]

"""Step 4d"""

for i in range(len(X)):

    if (max2 < e[i]):
        max2 = e[i]

        i2 = i

k = np.dot(X[i1], X[i1]) + np.dot(X[i2], X[i2]) + 2 * np.dot(X[i1], X[i2])  # 4f

old = alpha[i2]

alpha[i2] = alpha[i2] + (Y[i2] * e[i2] / k)  # Step 5

alpha[i1] = alpha[i1] + (Y[i1] * Y[i2]) * (old - alpha[i2])  # Step 6

for i in range(len(alpha)):

    if (alpha[i] < 0.0076):  # Step 7

        alpha[i] = 0

for i in range(len(alpha)):

    if (alpha[i] > 0):
        newi = i

        break

b = t1[newi] + 1 - Y[i] * np.dot(X[i, :], W)  # Step 8

predict = np.matmul(X, W) + b  # Step 9

predict = [1 if p > 0 else -1 for p in predict]

print("Predicted", predict)

print("Accuracy", sum([1 if i == j else 0 for i, j in zip(Y, predict)]) / len(Y))