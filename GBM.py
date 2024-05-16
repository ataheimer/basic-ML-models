import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve


pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore",category=Warning)

df = pd.read_csv('VS2.csv', sep=";")

df = df.apply(pd.to_numeric, errors='coerce')

df = df.fillna(0)

y = df["Olum"]

X = df.drop(["Olum", "Ad_Soyad", "Nuks_Cat", "preop_ast", "önemsizsırı", "ÖNEMSİZ"],axis=1)

gbm_model = GradientBoostingClassifier(random_state=42)

#print(gbm_model.get_params())

cv_results = cross_validate(gbm_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print(cv_results["test_accuracy"].mean())
#0.6551666666666666
print(cv_results["test_f1"].mean())
#0.7223609934690056
print(cv_results["test_roc_auc"].mean())
#0.7214867724867725


gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

print(gbm_best_grid.best_params_)

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=42).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print(cv_results["test_accuracy"].mean())
#0.6925
print(cv_results["test_f1"].mean())
#0.7613085246233188
print(cv_results["test_roc_auc"].mean())
#0.7291798941798942








