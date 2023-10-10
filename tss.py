import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_class_data(ytrain, xtrain):
    class_data = {}
    for cls in ytrain.unique():
        idx = ytrain[ytrain == cls].index
        class_data[cls] = xtrain.loc[idx, :]
    return class_data

def get_prior(ytrain):
    class_prior = {}
    for cls in ytrain.unique():
        class_prior[cls] = len(ytrain[ytrain == cls])/len(ytrain)
    return class_prior
def get_class_mean(class_data):

    class_mean = {}
    for cls in class_data:
        class_mean[cls] = np.array(class_data[cls].mean())
    return class_mean
def get_class_covariance(class_data):
    class_cov = {}
    class_cov_det = {}
    class_cov_inv = {}
    for cls in class_data:
        class_cov[cls] = np.cov(np.array(class_data[cls]).T)
        class_cov_det[cls] = np.linalg.det(class_cov[cls])
        class_cov_inv[cls] = np.linalg.inv(class_cov[cls])

    return class_cov, class_cov_inv, class_cov_det
def predict(x, class_mean, class_prior, class_data, class_cov_inv, class_cov_det):
    pos = [((-1/2)*(x - class_mean[cls]).T @ (class_cov_inv[cls]) @ (x - class_mean[cls])) + (-1/2)*np.log(class_cov_det[cls]) + np.log(class_prior[cls])
           for cls in class_data]
    ypred = np.argmax(pos)
    return ypred
def plot_decision_boundary(xtrain, ytrain, class_mean, class_prior, class_data,
                           class_cov_inv, class_cov_det, target_names):
    min1, max1 = xtrain.iloc[:, 0].min() - 1, xtrain.iloc[:, 0].max() + 1
    min2, max2 = xtrain.iloc[:, 1].min() - 1, xtrain.iloc[:, 1].max() + 1
    x1grid = np.arange(min1, max1, 0.01)
    x2grid = np.arange(min2, max2, 0.01)
    xx, yy = np.meshgrid(x1grid, x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1, r2))
    yhat = [predict(g, class_mean, class_prior, class_data, class_cov_inv, class_cov_det) for g in grid]
    zz = np.array(yhat).reshape(xx.shape)
    colors = ['r','y', 'b']
    markers = ['s','D', 'o']
    plt.figure(figsize=(10,8))
    plt.contourf(xx, yy, zz, cmap='viridis', alpha=0.8)
    for cls, color, mark, target_name in zip(class_data, colors,markers, target_names):
        idx = ytrain[ytrain == cls].index
        plt.scatter(xtrain.iloc[idx, 0], xtrain.iloc[idx, 1], alpha=0.8, color=color, label=target_name, marker=mark)
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        plt.title(" ")
    plt.show()
def train(ytrain, xtrain):
    cls_data = get_class_data(ytrain, xtrain)
    prior = get_prior(ytrain)
    class_mean = get_class_mean(cls_data)
    class_cov, class_cov_inv, class_cov_det = get_class_covariance(cls_data)
    return cls_data, prior, class_mean, class_cov, class_cov_inv, class_cov_det

def make_meshgrid(x, y, h=.02):
    d = 2
    x_min, x_max = x.min() - d, x.max() + d
    y_min, y_max = y.min() - d, y.max() + d
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
def file_loader():
    df = pd.read_csv("Iris.csv")
    return df

def correl_table(df, str_desc):
    columns = list(itertools.combinations((list(df.columns)), 2))
    figure, axis = plt.subplots(3, 2)
    figure.tight_layout()
    figure.canvas.manager.set_window_title(str_desc)
    axis = axis.ravel()
    for coordinates, col in zip(axis, columns):
        x, y = map(list, zip(*df.loc[:, col].values.reshape(-1, 2)))
        coordinates.scatter(x, y, 6.0, linewidths=0.5)
        coordinates.set_xlabel("Correlation is="+str(np.corrcoef(x, y)[0,1]))
        coordinates.set_title(col)
    plt.show()


class NaiveBayesClassifier:
    def __init__(self, class_num):
        self.class_num = class_num
        self.class_prior = [0] * class_num
        self.feature_probs = []

    def train(self, X_train, y_train):
        for y in y_train:
            self.class_prior[y] += 1

        for i in range(self.class_num):
            total_count = sum([1 for y in y_train if y == i])
            occurrence_count = [[0] * (np.max(X_train)+1) for _ in range(len(X_train[0]))]

            for j in range(len(X_train)):
                for k in range(len(X_train[j])):
                    if y_train[j] == i:
                        occurrence_count[k][X_train[j][k]] += 1

            feature_prob = [[0] * (np.max(X_train)+1) for _ in range(len(X_train[0]))]

            for j in range(len(occurrence_count)):
                for k in range(len(occurrence_count[j])):
                    feature_prob[j][k] = occurrence_count[j][k] / total_count

            self.feature_probs.append(feature_prob)

        total_samples = len(y_train)
        self.class_prior = [count / total_samples for count in self.class_prior]

    def predict(self, X_test):
        predictions = []

        for i in range(len(X_test)):
            max_prob = float('-inf')
            max_class = float('-inf')

            for j in range(self.class_num):
                prob = self.class_prior[j]

                for k in range(len(X_test[i])):
                    prob *= self.feature_probs[j][k][X_test[i][k]]

                if prob > max_prob:
                    max_prob = prob
                    max_class = j

            predictions.append(max_class)

        return predictions