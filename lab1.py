import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import svm
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tss import correl_table, file_loader, make_meshgrid, plot_contours, quadratic_linear_discriminant
from sklearn import datasets
import pandas as pd


def ex_1():
    df = file_loader().drop(["Id"], axis=1)

    label_encoder = preprocessing.LabelEncoder()
    df['Species'] = label_encoder.fit_transform(df['Species'])

    df = df.groupby("Species")
    species_list = list(itertools.chain.from_iterable(np.array(df["Species"].unique().to_list())))

    for i in range(len(species_list)):
        correl_table(df.get_group(species_list[i]), "Corelation between features for "+str(i))

    df = file_loader().drop(["Id"], axis=1)

    feature_names = df.columns[:-1]
    species_list = df["Species"].unique()

    figure, axis = plt.subplots(len(species_list), len(feature_names))
    axis = axis.ravel()

    figure.tight_layout()

    axis_cnt = 0
    for name in feature_names:
        for cls in species_list:
            axis[axis_cnt].hist(df.loc[df["Species"] == cls, name])
            axis[axis_cnt].set_xlabel(cls)
            axis[axis_cnt].set_ylabel(name)
            axis_cnt += 1

    axis.reshape(len(species_list), len(feature_names))
    plt.show()



def ex_2_file_loader():
    df = file_loader().drop(["Id"], axis=1)
    labels = df["Species"]
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    target_names = df["Species"].unique()
    df = df.drop("Species", axis=1)

    column = np.random.choice(df.columns, 2)
    features = np.array(df.loc[:, column])

    return labels, target_names, features, column, df
def ex_2():
    labels, target_names, features, column, _ = ex_2_file_loader()

    models = [svm.SVC(kernel='linear', C=1.0),
              MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 2), random_state=1),
              LinearDiscriminantAnalysis(),
              svm.SVC(kernel='poly', degree=2, gamma='auto', C=1.0),
              KNeighborsClassifier(n_neighbors=5),
              QuadraticDiscriminantAnalysis(),
              LogisticRegression()
              ]

    models = [clf.fit(features, labels) for clf in models]

    titles = ['SVM-linear',
              'MLP',
              'LinearDiscriminantAnalysis',
              'SVM-Quadratic',
              'QuadraticDiscriminantAnalysis',
              'LogisticRegression'
              ]

    fig, sub = plt.subplots(3, 2, figsize=(7, 2))

    X0, X1 = features[:, 0], features[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy, alpha=0.8)
        ax.scatter(X0, X1, c=labels, s=20, edgecolors='face')

        ax.set_xlabel(column[0])
        ax.set_ylabel(column[1])
        ax.set_title(title)

    plt.subplots_adjust(left=0.057, bottom=0.11, right=0.975, top=0.962, wspace=0.1, hspace=0.453)
    plt.show()

def ex_3():
    df = file_loader()
    df = df.drop("Id", axis=1)
    y = df["Species"]
    choosen_classses = np.random.choice(y.unique(), 2)
    y = pd.DataFrame(y)
    df = df[(df['Species'] == choosen_classses[0]) | (df['Species'] == choosen_classses[1])]
    filtered_y = y[y['Species'].isin(choosen_classses)]
    filtered_y = filtered_y.to_numpy()

    x = df.drop("Species", axis=1)

    model = LinearDiscriminantAnalysis()

    combinations = np.array(list(itertools.combinations(x.columns, 2)))
    fig, axes = plt.subplots(2, (combinations.shape[0]//2), figsize=(15, 6))
    for i, axis in enumerate(axes.flatten()):
        features = list(combinations[i])
        choosen_x = x[features].to_numpy()
        model = model.fit(choosen_x, filtered_y)
        predict = model.predict(choosen_x)

        for j in range(len(predict)):
            color = "pink"
            marker = ">"

            if predict[j] == choosen_classses[0]:
                color = "blue"
                marker = "o"
            if predict[j] == choosen_classses[1]:
                color = "red"
                marker = "o"
            elif predict[j] != filtered_y[j]:
                color = "black"
                marker = "*"
            axis.scatter(choosen_x[j, 0], choosen_x[j, 1], color=color, marker=marker)

    plt.show()


def ex_4():
    df = pd.read_csv("Iris.csv")
    selected_x = ["SepalLengthCm", "PetalWidthCm", "Species"]
    x = df[selected_x]
    cls_lst = df["Species"].unique()

    mean = x.groupby("Species").mean().to_numpy()
    cov = x.groupby("Species").cov().to_numpy().reshape(len(cls_lst), -1)

    classes = []
    for i in range(len(cls_lst)):
        classes.append([mean[i], cov[i].reshape(len(selected_x)-1, -1), np.linalg.inv(cov[i].reshape(len(selected_x)-1, -1))])

    linear_discriminant = []
    for i, cls in enumerate(cls_lst):
        species = x[x["Species"] == cls].drop("Species", axis=1)
        linear_discriminant.append(quadratic_linear_discriminant(species, classes[i][0], classes[i][1], classes[i][2]))




