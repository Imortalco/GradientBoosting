import sklearn
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': np.arange(100, 500, 100),
    'random_state': np.arange(1, 5, 1)
}

X = data.drop(["G3"], 1)
y = data["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gbr = GradientBoostingRegressor()

gbr_gs = GridSearchCV(gbr, grid, cv=4)
gbr_gs.fit(X_train, y_train)

print("Best Parameters:", gbr_gs.best_params_)
print("Train Score", gbr_gs.best_score_)
print("Test Score", gbr_gs.score(X_test, y_test))
