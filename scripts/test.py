import codecs
import io
import json
from multiprocessing import Pool
import os
import sys
import warnings
from traceback import format_exc

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score,recall_score, precision_score, precision_recall_curve, roc_auc_score
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
import pandas as pd
from patsy import dmatrices

warnings.filterwarnings("ignore")

sns.set(style = "ticks")
data_dir = "../data/"
input_dir = data_dir + "prediction_input/"
output_dir = data_dir + "simplified_prediction_output/"


models = {
    "SVM": SVC(kernel="rbf", C=1),
    "GBDT": GradientBoostingClassifier(
        max_depth=5, max_features="sqrt", n_estimators=125
    ),
    "Logistic": LogisticRegression(max_iter=1000),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(10, 30, 10),
        alpha=0.001,
        learning_rate="adaptive",
        max_iter=1000,
    ),
}


def train_model(df_ds, selected_columns):
    X, y = df_ds[selected_columns], df_ds["dropout"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_accuracies = {}
    model_aucs = {}
    model_pred_probs = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        if model_name == "SVM":
            y_pred_prob = model.decision_function(X_test_scaled)
        else:
            y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
        model_pred_probs[model_name] = list(y_pred_prob)
        model_aucs[model_name] = roc_auc_score(y_test, y_pred_prob)
        model_accuracies[model_name] = accuracy_score(y_test, y_pred)
        model_pred_probs["gt"] = list(y_test) #ground truth
    return model_accuracies, model_aucs, model_pred_probs


def train(measure, level, i):
    try:
        fouts = [open(f"{output_dir}/{measure}_{metric}_{level}_{i}", "w", buffering=1) 
                 for metric in ["accuracies", "auc", "pred"]]
        df_ds = list()
        for j in range(5):
            df_path = f"{input_dir}/{measure}_dropout_{level}_{i * 5 + j}"
            df_ds.append(pd.read_csv(df_path, dtype={"lang": "string"}, sep="\t", header=0))
        df_ds = pd.concat(df_ds)
        selected_columns = list(df_ds.columns)[1:-1]
        for fout, model_metric in zip(fouts, train_model(df_ds, selected_columns)):
            fout.write(f"{level}\t{i}\t{json.dumps(model_metric)}\n")
            fout.close()
    except Exception:
        print(f"Train and predict error {format_exc()}")


if __name__ == "__main__":
    measure_activity_levels = {"hour": [0.6], "day": [0.7]}
    with Pool(processes=12) as pool:
        for measure, levels in measure_activity_levels.items():
            for level in levels:
                for i in range(6):
                    pool.apply_async(train, args=(measure, level, i))
        pool.close()
        pool.join()