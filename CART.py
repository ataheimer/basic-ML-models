import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore",category=Warning)

df = pd.read_csv('VS2.csv', sep=";")

df = df.apply(pd.to_numeric, errors='coerce')

y = df["Olum"]

X = df.drop(["Olum", "Ad_Soyad", "Nuks_Cat", "preop_ast", "önemsizsırı", "ÖNEMSİZ"],axis=1)

cart_model = DecisionTreeClassifier(random_state=42).fit(X, y)

y_pred = cart_model.predict(X)

y_prob = cart_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
print(roc_auc_score(y, y_prob))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

cart_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

y_pred = cart_model.predict(X_test)

y_prob = cart_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("roc_auc")
print(roc_auc_score(y_test, y_prob))

cart_model = DecisionTreeClassifier(random_state=42).fit(X, y)

cv = cross_validate(cart_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print("\n\naccuracy")
print(cv["test_accuracy"].mean())
print("f1")
print(cv["test_f1"].mean())
print("roc_auc")
print(cv["test_roc_auc"].mean())

#print(cart_model.get_params())

cart_params = {'max_depth': range(1,11),
                'min_samples_split': range(2,20)}

cart_best_grid = GridSearchCV(cart_model, cart_params, scoring="f1", cv=10, n_jobs=-1, verbose=1).fit(X, y)

print(cart_best_grid.best_params_)

print(cart_best_grid.best_score_)

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)

cv = cross_validate(cart_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

print("\n\naccuracy")
print(cv["test_accuracy"].mean())
print("f1")
print(cv["test_f1"].mean())
print("roc_auc")
print(cv["test_roc_auc"].mean())

train_score, test_score = validation_curve(cart_final, X, y, param_name="max_depth", param_range=range(1,11), scoring="roc_auc", cv=10)

mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)
print(mean_train_score)
print(mean_test_score)


tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)

print(skompile(cart_final.predict).to("python/code"))

#print(skompile(cart_final.predict).to("sqlalchemy/sqlite"))

print(skompile(cart_final.predict).to("excel"))







print("son")