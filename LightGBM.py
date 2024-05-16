import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from lightgbm import LGBMClassifier


pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore",category=Warning)

df = pd.read_csv('VS2.csv', sep=";")

df = df.apply(pd.to_numeric, errors='coerce')

df = df.fillna(0)

y = df["Olum"]

X = df.drop(["Olum", "Ad_Soyad", "Nuks_Cat", "preop_ast", "önemsizsırı", "ÖNEMSİZ"],axis=1)

lightgbm_model = LGBMClassifier(random_state=42)

print(lightgbm_model.get_params())

cv_results = cross_validate(lightgbm_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print(cv_results["test_accuracy"].mean())
#0.6759999999999999
print(cv_results["test_f1"].mean())
#0.7386261383633463
print(cv_results["test_roc_auc"].mean())
#0.7044497354497354



lightgbm_params = {"learning_rate": [0.01, 0.1],
                "n_estimators": [100, 300, 500, 1000],
                "colsample_bytree": [0.5, 0.7, 1]}

lightgbm_best_grid = GridSearchCV(lightgbm_model, lightgbm_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

print(lightgbm_best_grid.best_params_)

lightgbm_final = lightgbm_model.set_params(**lightgbm_best_grid.best_params_, random_state=42).fit(X, y)

cv_results = cross_validate(lightgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print(cv_results["test_accuracy"].mean())
#0.6966666666666667
print(cv_results["test_f1"].mean())
#0.7582424916573972
print(cv_results["test_roc_auc"].mean())
#0.722984126984127


