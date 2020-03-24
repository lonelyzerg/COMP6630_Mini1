import pickle
import numpy as np
from libsvm.svmutil import svm_read_problem
from sklearn.model_selection import train_test_split

def select_feature(dataset, features):
    info_gains = [(info_gain(dataset, x), x) for x in features]
    return max(info_gains)[1]


def info_gain(dataset, feature):
    feature_values = set([x[feature] for x in dataset])
    entropy_sub_dataset = 0.0
    for val in feature_values:
        sub_dataset = [x for x in dataset if x[feature] == val]
        entropy_sub_dataset += float(len(sub_dataset)) / len(
            dataset) * entropy(sub_dataset)
    return entropy(dataset) - entropy_sub_dataset


def entropy(dataset):
    labels = [x[-1] for x in dataset]
    label_dict = {x: 0.0 for x in set(labels)}
    for label in labels:
        label_dict[label] += 1
    h = 0.0
    for k, v in label_dict.items():
        h -= v / len(labels) * np.log2(v / len(labels))
    return h


def majority_label(labels):
    label_dict = {x: 0 for x in set(labels)}
    for label in labels:
        label_dict[label] += 1
    return max(label_dict.items(), key=lambda x: x[1])[0]


def build_tree(dataset, features):
    labels = [x[-1] for x in dataset]
    if labels.count(labels[0]) == len(labels):
        return {'label': labels[0]}
    if len(features) == 0:
        return {'label': majority_label(labels)}
    best_feature = select_feature(dataset, features)
    tree = {'feature': best_feature, 'children': {}}
    best_feature_values = set([x[best_feature] for x in dataset])
    for val in best_feature_values:
        sub_dataset = list(filter(lambda x: x[best_feature] == val, dataset))
        if len(sub_dataset) == 0:
            tree['children'][val] = {
                'label': majority_label(labels)}
        else:
            tree['children'][val] = build_tree(
                sub_dataset, [x for x in features if x != best_feature])
    return tree


def predict(tree, sample_vector, default):
    if 'feature' in tree:
        try:
            return predict(tree['children'][sample_vector[tree['feature']]], sample_vector, default)
        except:
            return default
    else:
        return tree['label']


y_raw, x_raw = svm_read_problem('a4a')

dataset = np.zeros((len(y_raw), 124))

for i in range(len(y_raw)):
    line = x_raw[i]
    for k, v in line.items():
        dataset[i][k - 1] = v

dataset[:, 123] = np.array(y_raw)
features = [x for x in range(123)]

tree = build_tree(dataset, features)
if(2 * np.sum(dataset[:,-1]==-1) > len(dataset[:,-1])):
    default = -1
else:
    default = 1

y_test, x_test = svm_read_problem('a4a.t')
dataset_test = np.zeros((len(y_test), 124))

for i in range(len(y_test)):
    line = x_test[i]
    for k, v in line.items():
        dataset_test[i][k - 1] = v
dataset_test[:, 123] = np.array(y_test)

predicted = np.zeros(len(y_test))
for i in range(len(y_test)):
    predicted[i] = predict(tree, dataset_test[i], default)

print('\n======= a4a =======')
print(len(predicted), 'test cases predicted.')
correct_num = np.sum(predicted == dataset_test[:, -1])
print(correct_num, 'are correct.')
print('Accuracy = ', np.round(correct_num * 100 / len(predicted)), '%', sep='')





print('\n======= iris =======')

y_raw, x_raw = svm_read_problem('iris.scale')

y = np.array(y_raw)
x = np.zeros((len(y_raw), 4))
for i in range(len(y_raw)):
    line = x_raw[i]
    for k, v in line.items():
        x[i][k - 1] = v

x[x<-0.5]=-2
x[ np.logical_and(x<0, x>=-0.5) ] = -1
x[np.logical_and(x>=0, x<=0.5)] = 1
x[x>0.5] = 2

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

dataset = np.zeros((len(y_train), 5))
dataset[:, [0,1,2,3]] = x_train
dataset[:, 4] = y_train

dataset_test = np.zeros((len(y_test), 5))
dataset_test[:, [0,1,2,3]] = x_test
dataset_test[:, 4] = y_test

features = [x for x in range(4)]

tree = build_tree(dataset, features)

predicted = np.zeros(len(y_test))
for i in range(len(y_test)):
    predicted[i] = predict(tree, dataset_test[i], default)

print(len(predicted), 'test cases predicted.')
correct_num = np.sum(predicted == dataset_test[:, -1])
print(correct_num, 'are correct.')
print('Accuracy = ', np.round(correct_num * 100 / len(predicted)), '%', sep='')