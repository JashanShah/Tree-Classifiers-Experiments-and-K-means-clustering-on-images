import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

path = r"C:\HW3\hw3_part1_data\all_data"
firstNum = [3, 5, 10, 15, 18]
secNum = [1, 10, 50]

def BaggingClassifier():
    XValid= validData.iloc[:, 0:-1].values
    YValid= validData.iloc[:, -1].values
    XTest= testData.iloc[:, 0:-1].values
    YTest= testData.iloc[:, -1].values
    parameters = {'n_estimators' : [10,50,70,100,120],'max_samples' : [0.05, 0.1, 0.2, 0.5, 0.7, 1.0],'max_features' : [0.05, 0.1, 0.2, 0.5, 1.0]}
    grid = GridSearchCV(BaggingClassifier(DecisionTreeClassifier()), parameters)
    grid.fit(XValid, YValid)
    print("\nThe Tuned Hyperparameters : ", grid.best_params_)
    print("The Accuracy on Validation: ", grid.best_score_)

    trainX1 = trainData.iloc[:, 0:-1]
    validX1 = validData.iloc[:, 0:-1]

    #Combining training and validation features
    XData = pd.concat([trainX1, validX1], axis=0, copy=True).values

    XData.shape


    trainY1 = trainData.iloc[:, -1]
    validY2 = validData.iloc[:, -1]

    YData = pd.concat([trainY1, validY2], axis=0, copy=True).values

    tree = BaggingClassifier(DecisionTreeClassifier(), **grid.best_params_)
    tree = tree.fit(XData, YData)

    yPredict = tree.predict(XTest)

    print("\n\nAccuracy on Testing: ", metrics.accuracy_score(YTest, yPredict))
    print("F1 Score: ", f1_score(YTest, yPredict))
    print("----------------------------------------------------------------------------------------")


for i in firstNum:
  for j in secNum:
    trainData = pd.read_csv(path + "/train_c" + str(i) + "00_d" + str(j) + "00.csv", header=None)
    validData = pd.read_csv(path + "/valid_c" + str(i) + "00_d" + str(j) + "00.csv", header=None)
    testData = pd.read_csv(path + "/test_c" + str(i) + "00_d" + str(j) + "00.csv", header=None)

    print("\nData Set Name: " + str(i) + "00_d" + str(j) + "00\n")
    BaggingClassifier()
