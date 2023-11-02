import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.decomposition import PCA
import numpy as np

def task_1_2():
    df = pd.read_csv('car_ad.csv', encoding ='iso-8859-9')
    df = df[0:500]
    y = df["car"]
    x = df[["price", "mileage", "engV"]].fillna(0)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    fe = preprocessing.LabelEncoder()
    x = x.apply(fe.fit_transform)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression()
    gb_model = GradientBoostingClassifier(n_estimators=82, learning_rate=1.0, max_depth=1, random_state=0)

    lr_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    lrpred = lr_model.predict(X_test)
    gbpred = gb_model.predict(X_test)

    df = {"label": y_test, "lr_predict": lrpred, "gb_predict": gbpred}
    df = pd.DataFrame(data=df)
    print(df)


def task_3():
    df = pd.read_csv('car_ad.csv', encoding='iso-8859-9')
    x = df[['body', 'engType', 'model']].fillna(0)
    y = df['registration']

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    ohot = preprocessing.OneHotEncoder()
    targ_encod = preprocessing.TargetEncoder()

    y_hot = ohot.fit_transform(x, y)
    y_targ = targ_encod.fit_transform(x, y)

    print(y_hot, y_targ)


def task_4():
    df = pd.read_csv('car_ad.csv', encoding='iso-8859-9')

    y = df["car"]
    x = df[["price", "mileage", "engV"]].fillna(0)

    pcamodel = PCA(n_components=2)
    pca = pcamodel.fit_transform(x)

    plt.bar(range(1, len(pcamodel.explained_variance_) + 1), pcamodel.explained_variance_)
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.plot(range(1, len(pcamodel.explained_variance_) + 1),
             np.cumsum(pcamodel.explained_variance_),
             c='red',
             label="Cumulative Explained Variance")
    plt.legend(loc='upper left')
    plt.show()

task_4()
