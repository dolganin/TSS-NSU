import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import model_selection
from tss import NaiveBayesClassifier
from sklearn.naive_bayes import GaussianNB

def task_1():
    df = pd.read_csv('mushrooms.csv')
    y = df["class"]
    for feature in df.columns[1:]:
        df.groupby(['class', feature]).size().reset_index(name='count').fillna(0).plot.bar(figsize=(10, 7))
    plt.show()

def task_2():
    df = pd.read_csv('mushrooms.csv')
    label_encoder = preprocessing.LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])

    variables = df.columns[1:]
    best_variable = None
    min_errors = float('inf')
    for variable in variables:
        freq_table = df.groupby([variable, 'class']).size().unstack(fill_value=0)
        decision_function = freq_table.idxmax(axis=1)
        predictions = decision_function[df[variable]]
        errors = [0 if df['class'][i] == predictions[i] else predictions[i] for i in range(len(predictions))]
        error_count = sum(errors)

        if error_count < min_errors:
            min_errors = error_count
            best_variable = variable

    accuracy_scores = {}
    for variable in variables:
        freq_table = df.groupby([variable, 'class']).size().unstack(fill_value=0)
        decision_function = freq_table.idxmax(axis=1)
        predictions = decision_function[df[variable]]
        accuracy = accuracy_score(df['class'], predictions)
        accuracy_scores[variable] = accuracy


    print("The best feature for calcualtion is =", variables[(list(accuracy_scores.values()).index(max(accuracy_scores.values())))],
          "with accuracy equals to =", (max(accuracy_scores.values())))
def task_3():
    df = pd.read_csv('mushrooms.csv')
    label_encoder = preprocessing.LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])

    y = df['class']
    X = df.drop('class', axis=1)
    le = preprocessing.LabelEncoder()
    X = X.apply(le.fit_transform)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=42)

    gauss_model = GaussianNB()

    gauss_model.fit(X_train, y_train)
    print("Acc: {}%".format(round(gauss_model.score(X, y) * 100)))


def task_4():
    NBC = NaiveBayesClassifier(2)

    df = pd.read_csv('mushrooms.csv')
    label_encoder = preprocessing.LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])

    y = df['class']
    X = df.drop('class', axis=1)
    le = preprocessing.LabelEncoder()
    X = X.apply(le.fit_transform)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=42)
    X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    NBC.train(X_train, y_train)
    predict = NBC.predict(X_test)
    print(round(accuracy_score(predict, y_test), 3))




def task_5():
    data = pd.read_csv('mushrooms.csv')

    label_encoder = preprocessing.LabelEncoder()
    for column in data.columns:
        data[column] = label_encoder.fit_transform(data[column])

    X = data.drop('class', axis=1)
    y = data['class']
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)

    inverse_transform = np.exp(probabilities) / np.sum(np.exp(probabilities), axis=1).reshape(-1, 1)
    print(inverse_transform)
