import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings("ignore")

#Getting Data
# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.
# rescale the data, use the traditional train/test split

# (60K: Train) and (10K: Test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

#Decision tree
Dtree = DecisionTreeClassifier()
Dtree = Dtree.fit(X_train, y_train)
Ypred1 = Dtree.predict(X_test)
print("\nThe accuracy of DT: ", metrics.accuracy_score(y_test, Ypred1))

#Bagging
Bagging = BaggingClassifier()
Bagging = Bagging.fit(X_train,y_train)
Ypred2 = Bagging.predict(X_test)
print("\nThe accuracy of Bagging: ", metrics.accuracy_score(y_test, Ypred2))

#Random Forest
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train, y_train)
Ypred3 = RandomForest.predict(X_test)
print("\nThe accuracy of Random Forest: ", metrics.accuracy_score(y_test, Ypred3))

#Gradient Boosting
GradientBoosting = GradientBoostingClassifier()
GradientBoosting = GradientBoosting.fit(X_train, y_train)
Ypred4 = GradientBoosting.predict(X_test)
print("\nThe accuracy of Boosting: ", metrics.accuracy_score(y_test, Ypred4))
