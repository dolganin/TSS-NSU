import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn import  model_selection

def task_1_2():
    df = pd.read_csv('car_ad.csv', encoding ='iso-8859-9')
    y = df["car"]
    x = df[["price", "mileage", "engV"]].fillna(0)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    fe = preprocessing.LabelEncoder()
    x = x.apply(fe.fit_transform)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression()
    gb_model = GradientBoostingClassifier(n_estimators=14, learning_rate=1.0, max_depth=1, random_state=0)

    lr_model.fit(X_test, y_test)
    gb_model.fit(X_test, y_test)



def task_3():
    df = pd.read_csv('car_ad.csv', encoding='iso-8859-9')
    y = df["car"]
    x = df.drop("car", axis=1)

    ohot = preprocessing.OneHotEncoder()
    targ_encod = preprocessing.TargetEncoder()

    y_hot = ohot.fit_transform(y)
    y_targ = targ_encod.fit_transform(y)

    x_hot = x.apply(ohot.fit_transform)
    x_targ = x.apply(targ_encod.fit_transform)



