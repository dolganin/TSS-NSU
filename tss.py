import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def quadratic_linear_discriminant(x, mean, cov, inv_cov):
   return -0.5 * (x - mean).to_numpy() @ inv_cov @ (x - mean).T - 0.5 * np.log(np.linalg.det(cov)) + np.log(0.5)