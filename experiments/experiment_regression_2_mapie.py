"""Benchmarking run time of MapieRegressor + RandomForestRegressor"""

from mapie.regression import MapieRegressor
from mapie.subsample import Subsample
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv
import os
import time
from sklearn.datasets import fetch_california_housing

from urllib.request import urlretrieve
import zipfile

result_folder = "results_regression_time"

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

rng = np.random.default_rng(2)
randints = rng.integers(0, high=1e6, size=50)
alpha = 0.05


################################
# Training & Evaluation workflow
################################


def experiment(data_name, X_train, y_train, X_test, y_test, method_list, params):
    for method_name, method, cv in method_list:
        echo = []
        train_times = []
        test_times = []

        for k in range(30):
            if name == "bootstrap":
                estimator = DecisionTreeRegressor(max_features="sqrt")
            else:
                estimator = RandomForestRegressor(
                    n_estimators=params[name]["n_estimators"]
                )
            rfreg = MapieRegressor(
                estimator=estimator,
                method=method,
                cv=cv,
                test_size=0.3,
                random_state=randints[k],
                n_jobs=-1,
            )
            curr1 = time.time()
            rfreg.fit(X_train, y_train)
            curr2 = time.time()
            curr3 = time.time()
            _, intervals = rfreg.predict(X_test, alpha=alpha, ensemble=True)
            curr4 = time.time()
            intervals = np.squeeze(intervals)
            train_times.append(curr2 - curr1)
            test_times.append(curr4 - curr3)
            print("time", curr2 - curr1, curr4 - curr3)
            echo.append(k + 1)
            print("echo", k, ", alpha =", alpha)

        filename = f"{data_name}_{method_name}_MAPIE.csv"
        with open(filename, "w", newline="") as f:
            write = csv.writer(os.path.join(result_folder, f))
            write.writerows([echo, echo, echo, train_times, test_times])


# Set up methods and parameters.
method_list = [
    ("bootstrap", "plus", Subsample(n_resamplings=int(1000 * 0.9))),
    ("cv", "plus", 10),
    ("split", "base", "split"),
]

param_dict = {"cv": {"n_estimators": 100}, "split": {"n_estimators": int(1000 * 0.9)}}


#################
# Housing dataset
#################

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

experiment("HousingData", X_train, y_train, X_test, y_test, method_list, param_dict)


##################
# Concrete dataset
##################

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
data = pd.read_excel(url)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

experiment("ConcreteData", X_train, y_train, X_test, y_test, method_list, param_dict)


##############
# Bike dataset
##############

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"

urlretrieve(url, "bike_sharing.zip")
with zipfile.ZipFile("bike_sharing.zip", "r") as zip_ref:
    zip_ref.extractall("bike_data")

data = pd.read_csv("bike_data/hour.csv")

X = data.drop(["instant", "dteday", "casual", "registered", "cnt"], axis=1)
y = data["cnt"]  # total count as target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

experiment("BikeData", X_train, y_train, X_test, y_test, method_list, param_dict)


###############
# Crime dataset
###############

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"

data = pd.read_csv("communities.data", header=None, na_values=["?"])

X = data.iloc[:, 5:-1]
y = data.iloc[:, -1]  # violent crimes per population

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

experiment("CrimeData", X_train, y_train, X_test, y_test, method_list, param_dict)
