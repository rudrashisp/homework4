import random, numpy as np
from matplotlib import pyplot
from numpy import meshgrid

random.seed(111)
X = [[random.randint(-40, 40), random.randint(-40, 40), 1] for i in range(200)]
Y = [1 if i + 3 * j > 2 else -1 for i, j, k in X]
xpos, xneg = [k for k in zip(*[i for i, j in zip(X, Y) if j == 1])], [k for k in
                                                                      zip(*[i for i, j in zip(X, Y) if j == -1])]
print("count of 1(s) >>", Y.count(1), "| count of -1(s)>>", Y.count(-1))


def fn_delta(eta, epoch):
    W, acc_list, error_list = [random.random() for i in range(3)], [], []
    for i in range(epoch):
        o = [sum([i * j for i, j in zip(x, W)]) for x in X]
        g = [eta * np.power(0.8, epoch + 1) * (i - j) for i, j in zip(Y, o)]
        W = [[i * k for k in j] for i, j in zip(g, X)] + [W]
        W = [sum(i) for i in zip(*W)]
        o = [1 if sum([i * j for i, j in zip(x, W)]) > 0 else -1 for x in X]
        acc = sum([i == j for i, j in zip(Y, o)]) / len(X)
        print(i + 1, '>> accuracy >> ', acc)
        acc_list += [acc]
        error_list += [1 - acc]
    t = [i + 1 for i in range(epoch)]
    pyplot.plot(t, error_list)
    pyplot.show()
    return o, acc_list, W


def fn_decision_boundary(wt):
    x1, x2 = range(-41, 41), range(-41, 41)
    z = [1 if i * wt[0] + j * wt[1] + wt[2] > 0 else -1 for i in x1 for j in x2]
    xx, yy = meshgrid(x1, x2)
    zz = np.array(z).reshape(xx.shape)
    pyplot.contourf(xx, yy, zz, cmap='Paired')
    pyplot.scatter(xpos[0], xpos[1], cmap='Paired')
    pyplot.scatter(xneg[0], xneg[1], cmap='Paired')

pyplot.xlabel("number of epochs")
pyplot.ylabel("Train Error")
pyplot.title('Variable learning rate')

pred, acc, final_wt = fn_delta(eta=0.01, epoch=25)
fn_decision_boundary(final_wt)