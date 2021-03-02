import pandas as pd
import numpy as np
from ders.data_prep import *
from ders.eda import *
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, GridSearchCV

pd.set_option ("display.max_columns", None)
pd.set_option ('display.expand_frame_repr', False)
pd.set_option ('display.width', 1000)

df_ = pd.read_csv ("/Users/ozanguner/PycharmProjects/son_dsmlbc/datasets/diabetes (1).csv")

df = df_.copy ()


def diabetes_prep(data):
    dataframe = data.copy ()
    zero_columns = [col for col in dataframe.columns if (dataframe[col].min () == 0) & (col != "Pregnancies")]
    for col in zero_columns:
        dataframe[col] = dataframe[col].apply (lambda x: dataframe[col].median () if x == 0 else x)

    # Glucose
    dataframe["NEW_GLUCOSE_CAT"] = pd.cut (dataframe["Glucose"],
                                           bins=[0, 100, 140, 199, dataframe["Glucose"].max () + 1],
                                           labels=["LOW", "NORMAL", "PREDIABETES", "T2_DIABETES"])

    # BMI
    dataframe["NEW_BMI_CAT"] = pd.cut (dataframe["BMI"], bins=[0, 18, 20, 25, 30, dataframe["BMI"].max () + 1],
                                       labels=["TOO_UNDERWEIGHT", "UNDERWEIGHT", "HEALTHY", "OVERWEIGHT", "OBESE"])

    # SkinThickness
    dataframe["NEW_SKINTHICKNESS_CAT"] = pd.cut (dataframe["SkinThickness"],
                                                 bins=[0, 15, 30, dataframe["SkinThickness"].max () + 1],
                                                 labels=["LEAN_BODY", "NORMAL", "FAT_BODY"])

    # Insulin
    dataframe["NEW_INSULIN_CAT"] = pd.cut (dataframe["Insulin"], bins=[0, 16, 166, dataframe["Insulin"].max () + 1],
                                           labels=["LOW", "NORMAL", "HIGH"])

    # BloodPressure
    dataframe["NEW_BLOODP_CAT"] = pd.cut (dataframe["BloodPressure"],
                                          bins=[0, 80, 90, 120, dataframe["BloodPressure"].max () + 1],
                                          labels=["NORMAL", "HYPERTENSION_1", "HYPERTENSION_2", "HYPERTENSIVE_CRISIS"])

    # Classification the "Age" Variable
    dataframe["NEW_AGE_CAT"] = pd.cut (dataframe["Age"],
                                       bins=[20, 35, 55, dataframe["Age"].max ()],
                                       labels=["YOUNG_ADULT", "ADULT", "SENIOR"])

    # Assigning numeric values to categorical variables
    dataframe["NEW_GLUCOSE_INT"] = dataframe["NEW_GLUCOSE_CAT"].map (
        {"LOW": 1, "NORMAL": 2, "PREDIABETES": 3, "T2_DIABETES": 4}).astype (int)

    dataframe["NEW_BMI_INT"] = dataframe["NEW_BMI_CAT"].map (
        {"TOO_UNDERWEIGHT": 1, "UNDERWEIGHT": 2, "HEALTHY": 3, "OVERWEIGHT": 4, "OBESE": 5}).astype (int)

    dataframe["NEW_SKINTHICKNESS_INT"] = dataframe["NEW_SKINTHICKNESS_CAT"].map (
        {"LEAN_BODY": 1, "NORMAL": 2, "FAT_BODY": 3}).astype (int)

    dataframe["NEW_INSULIN_INT"] = dataframe["NEW_INSULIN_CAT"].map ({"LOW": 1, "NORMAL": 2, "HIGH": 3}).astype (int)

    dataframe["NEW_BLOODP_INT"] = dataframe["NEW_BLOODP_CAT"].map (
        {"NORMAL": 1, "HYPERTENSION_1": 2, "HYPERTENSION_2": 3, "HYPERTENSIVE_CRISIS": 4}).astype (int)

    dataframe["NEW_AGE_INT"] = dataframe["NEW_AGE_CAT"].map ({"YOUNG_ADULT": 1, "ADULT": 2, "SENIOR": 3}).astype (int)

    dataframe["BMI_ST"] = dataframe["NEW_BMI_INT"] * dataframe["NEW_SKINTHICKNESS_INT"]
    dataframe["BP_AGE"] = dataframe["NEW_BLOODP_INT"] * dataframe["NEW_AGE_INT"]
    dataframe["GL_INS"] = dataframe["NEW_GLUCOSE_INT"] * dataframe["NEW_INSULIN_INT"]

    return dataframe


df = diabetes_prep (df)

df.isnull ().sum ()

df.describe ().T

df.info ()

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

df_new = df[num_cols]

# Examining Outliers and removing them
lof = LocalOutlierFactor (n_neighbors=20)
lof.fit_predict (df_new)
scores = lof.negative_outlier_factor_

np.sort (scores)

threshold_value = np.sort (scores)[5]
df_model = df_new[scores >= threshold_value]

# MACHINE LEARNING
X = df_new.drop ("Outcome", axis=1)
y = df_new["Outcome"]

X_train, X_test, y_train, y_test = train_test_split (X, y, random_state=17, test_size=0.2)

cart = DecisionTreeClassifier (random_state=17)
cart_model = cart.fit (X_train, y_train)
y_pred = cart_model.predict (X_test)
accuracy_score (y_test, y_pred)
print (classification_report (y_test, y_pred))
# recall => 0.54

y_prob = cart_model.predict_proba (X_test)[:, 1]
roc_auc_score (y_test, y_prob)
# AUC => 0.6946

# MODEL TUNING
cart_params = {'max_depth': [1, 2, 3, 4, 5, 6, 10, None],
               "min_samples_split": [2, 3, 5, 8],
               "max_features": [3, 5, 10, 15],
               "class_weight": ["balanced", None]}
cart_cv = GridSearchCV (cart_model, cart_params, cv=5, n_jobs=-1, verbose=True).fit (X_train, y_train)

cart_cv.best_params_

# FINAL MODEL
tuned_cart = DecisionTreeClassifier (random_state=17, **cart_cv.best_params_).fit (X_train, y_train)
tuned_pred = tuned_cart.predict (X_test)
accuracy_score (y_test, tuned_pred)
print (classification_report (y_test, tuned_pred))
# recall => 0.75

tuned_prob = tuned_cart.predict_proba (X_test)[:, 1]
roc_auc_score (y_test, tuned_prob)
# AUC => 0.8125


def plot_importance(model, features, num=len (X), save=False):
    feature_imp = pd.DataFrame ({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure (figsize=(10, 10))
    sns.set (font_scale=1)
    sns.barplot (x="Value", y="Feature", data=feature_imp.sort_values (by="Value",
                                                                       ascending=False)[0:num])
    plt.title ('Features')
    plt.tight_layout ()
    plt.show ()
    if save:
        plt.savefig ('importances.png')


plot_importance (tuned_cart, X_train)

# GL_INS is the most important feature
