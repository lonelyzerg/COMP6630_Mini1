import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from libsvm.svmutil import svm_read_problem
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def compute_cost(W, X, Y):
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0
    hinge_loss = regularization_strength * (np.sum(distances) / N)

    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

def calculate_cost_gradient(W, X_batch, Y_batch):
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])

    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (regularization_strength * Y_batch[ind] * X_batch[ind])
        dw += di

    dw = dw/len(Y_batch)
    return dw

def sgd(features, outputs):
    max_iters = 1000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.001
    for iter in range(1, max_iters):
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)

        if iter == 2 ** nth or iter == max_iters - 1:
            cost = compute_cost(weights, features, outputs)
            #print("iter {} Cost = {}".format(iter, cost))
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1

    return weights


regularization_strength = 10
learning_rate = 0.00001

y_raw, x_raw = svm_read_problem('a4a')

dataset = np.zeros((len(y_raw), 123))

for i in range(len(y_raw)):
    line = x_raw[i]
    for k, v in line.items():
        dataset[i][k - 1] = v

y = np.array(y_raw)

W = sgd(dataset, y)


y_test, x_test = svm_read_problem('a4a.t')
dataset_test = np.zeros((len(y_test), 123))

for i in range(len(y_test)):
    line = x_test[i]
    for k, v in line.items():
        dataset_test[i][k - 1] = v

y_test = np.array(y_test)

predicted = np.zeros(dataset_test.shape[0])
for i in range(dataset_test.shape[0]):
    predicted[i] = np.sign(np.dot(dataset_test[i], W))

print('\n======= a4a =======')
print(len(predicted), 'test case predicted.')
correct_num = np.sum(predicted == y_test)
print(correct_num, ' are correct.', sep='')
print('Accuracy = ', np.round(correct_num * 100 / len(predicted)), '%', sep='')

top_index = np.argsort(-np.abs(W))[:20]
plt.figure()
plt.bar(list(map(str,top_index)), W[top_index])
plt.xlabel('Feature Number')
plt.ylabel('Weight')
plt.title('Top 20 Weights in a4a Dataset')
plt.show(block=False)

print('\n======= iris =======')

y_raw, x_raw = svm_read_problem('iris.scale')

y = np.array(y_raw)
x = np.zeros((len(y_raw), 4))
for i in range(len(y_raw)):
    line = x_raw[i]
    for k, v in line.items():
        x[i][k - 1] = v

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

y_train1 = np.copy(y_train)
y_train1[y_train1!=1] = -1
y_train1[y_train1!=-1] = 1

y_train2 = np.copy(y_train)
y_train2[y_train2!=2] = -1
y_train2[y_train2!=-1] = 1

y_train3 = np.copy(y_train)
y_train3[y_train3!=3] = -1
y_train3[y_train3!=-1] = 1

W1 = sgd(x_train, y_train1)
W2 = sgd(x_train, y_train2)
W3 = sgd(x_train, y_train3)


predicted = np.zeros(len(y_test))
for i in range(len(y_test)):
    pred = np.zeros(3)

    pred[0] = np.dot(x_test[i], W1)
    pred[1] = np.dot(x_test[i], W2)
    pred[2] = np.dot(x_test[i], W3)
    predicted[i] = np.argmax(pred) + 1

print(len(predicted), ' test case predicted.', sep='')
correct_num = np.sum(predicted == y_test)
print(correct_num, ' are correct.', sep='')
print('Accuracy = ', np.round(correct_num * 100 / len(predicted)), '%', sep='')


top_index = np.argsort(-np.abs(W1))
plt.figure()
plt.bar(list(map(str,top_index)), W1[top_index])
plt.xlabel('Feature Number')
plt.ylabel('Weight')
plt.title('Weights in iris Dataset (Class 1)')
plt.show(block=False)

top_index = np.argsort(-np.abs(W2))
plt.figure()
plt.bar(list(map(str,top_index)), W2[top_index])
plt.xlabel('Feature Number')
plt.ylabel('Weight')
plt.title('Weights in a4a Dataset (Class 2)')
plt.show(block=False)

top_index = np.argsort(-np.abs(W3))
plt.figure()
plt.bar(list(map(str,top_index)), W3[top_index])
plt.xlabel('Feature Number')
plt.ylabel('Weight')
plt.title('Weights in a4a Dataset (Class 3)')
plt.show()
