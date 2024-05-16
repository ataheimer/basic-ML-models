import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from xgboost import XGBClassifier

pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore",category=Warning)

df = pd.read_csv('VS2.csv', sep=";")

df = df.apply(pd.to_numeric, errors='coerce')

df = df.fillna(0)

y = df["Olum"]

X = df.drop(["Olum", "Ad_Soyad", "Nuks_Cat", "preop_ast", "önemsizsırı", "ÖNEMSİZ"],axis=1)

xgboost_model = XGBClassifier(random_state=42)

#cv_results = cross_validate(xgboost_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

#print(cv_results["test_accuracy"].mean())
#0.65
#print(cv_results["test_f1"].mean())
#0.71
#print(cv_results["test_roc_auc"].mean())
#0.69


#print(xgboost_model.get_params())

xgboost_params = {"learning_rate": [0.17, 0.2, 0.23],
              "max_depth": [1, 7, 13],
              "n_estimators": [30, 100, 170],
              "colsample_bytree": [0.05, 0.7, 1.5]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

print(xgboost_best_grid.best_params_)

gbm_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=42).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print(cv_results["test_accuracy"].mean())

print(cv_results["test_f1"].mean())

print(cv_results["test_roc_auc"].mean())

