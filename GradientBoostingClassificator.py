import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
clas = le.fit_transform(list(data["class"]))

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(clas)

grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': np.arange(100, 500, 100),
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gbc = GradientBoostingClassifier()

gbc_gs = GridSearchCV(gbc, grid, cv=4)
gbc_gs.fit(X_train, y_train)

print("Best Parameters:", gbc_gs.best_params_)
print("Train Score:", gbc_gs.best_score_)
print("Test Score:", gbc_gs.score(X_test, y_test))

"""gbc.fit(X_train, y_train)
print("Accuracy is:", gbc.score(X_test, y_test))"""
