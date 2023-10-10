from sklearn import tree
import pandas as pd
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn import ensemble
from tss import lab_3_file_loader

def task_1():
    X_train, X_test, y_train, y_test = lab_3_file_loader()

    model = tree.DecisionTreeClassifier(criterion="entropy")
    model.fit(X_train, y_train)

    print(model.score(X_test, y_test))

    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(model,
                       feature_names=x.columns.tolist(),
                       class_names=["Healthy", "Disease"],
                       filled=True)

    plt.show()

def task_2():
    X_train, X_test, y_train, y_test = lab_3_file_loader()

    model = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    model.fit(X_train, y_train)

    print(model.score(X_test, y_test))

    importance = model.feature_importances_
    for i, feature in enumerate(x.columns):
        print(f"Variable {feature} importance: {importance[i]}")
    # Получение списка построенных деревьев
    trees = model.estimators_
    for i, tree in enumerate(trees):
        print(f"Tree {i + 1}:\n{tree}")

def task_3():
    X_train, X_test, y_train, y_test = lab_3_file_loader()
    x = []
    y = []
    for i in range(2, 100):
        model = tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=i)
        model.fit(X_train, y_train)
        x.append(i)
        y.append(model.score(X_test, y_test))

    plt.plot(x, y)
    plt.show()

def task_4():
    X_train, X_test, y_train, y_test = lab_3_file_loader()
    x = []
    y = []
    for i in range(2, 100):
        model = ensemble.GradientBoostingClassifier(n_estimators=i, learning_rate=1.0, max_depth=1, random_state=0)
        model.fit(X_train, y_train)
        x.append(i)
        y.append(model.score(X_test, y_test))

    plt.plot(x, y)
    plt.show()

def task_5():
    X_train, X_test, y_train, y_test = lab_3_file_loader()
    x = []
    y = []
    for i in range(2, 100):
        model = ensemble.RandomForestClassifier(n_estimators=i)
        model.fit(X_train, y_train)
        x.append(i)
        y.append(model.score(X_test, y_test))

    plt.plot(x, y)
    plt.show()

