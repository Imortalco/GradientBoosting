import pandas as pd
import sklearn
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

"""X, Y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)

model = GradientBoostingClassifier()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=1)

print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))"""

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': np.arange(100, 500, 100),
}

X = data.drop(["G3"], 1)
y = data["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gbc = GradientBoostingClassifier()

gbc_gs = GridSearchCV(gbc, grid, cv=4)
gbc_gs.fit(X_train, y_train)

print("Best Parameters:", gbc_gs.best_params_)
print("Train Score", gbc_gs.best_score_)
print("Test Score", gbc_gs.score(X_test, y_test))
