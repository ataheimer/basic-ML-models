import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore",category=Warning)

df = pd.read_csv('VS2.csv', sep=";")

df = df.apply(pd.to_numeric, errors='coerce')

df = df.fillna(0)

y = df["Olum"]

X = df.drop(["Olum", "Ad_Soyad", "Nuks_Cat", "preop_ast", "önemsizsırı", "ÖNEMSİZ"],axis=1)

rf_model = RandomForestClassifier(random_state=42)

#print(rf_model.get_params())

nan_columns = X.columns[X.isna().any()].tolist()
print("Columns with NaN values:", nan_columns)

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print(cv_results["test_accuracy"].mean())
print(cv_results["test_f1"].mean())
print(cv_results["test_roc_auc"].mean())









print(".")