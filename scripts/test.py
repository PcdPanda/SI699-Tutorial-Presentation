import json

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def train_model(df_ds, selected_columns):
    X = df_ds[selected_columns]
    y = df_ds["dropout"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_accuracies = {}
    model_aucs = {}
    model_pred_probs = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[model_name] = accuracy
        if model_name == "SVM1":
            y_pred_prob = model.decision_function(X_test_scaled)
        else:
            y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
        model_pred_probs[model_name] = list(y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)
        model_aucs[model_name] = auc
        model_pred_probs["gt"] = list(y_test) #ground truth
    return model_accuracies, model_aucs, model_pred_probs

if __name__ == "__main__":
    data_dir = "../data/"
    input_dir = data_dir + "/prediction_input"
    output_dir = data_dir + "/prediction_output"
    measure_activity_levels = {"hour": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                               "day": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
    models = {"SVM1": SVC(kernel="rbf", C=1),
            "GBDT": GradientBoostingClassifier(max_depth=5, max_features="sqrt", n_estimators=125),
            "logistic": LogisticRegression(max_iter=1000),
            "MLP": MLPClassifier(hidden_layer_sizes=(10, 30, 10), alpha=0.001, learning_rate="adaptive", max_iter=1000)
            }

    for measure in measure_activity_levels:
        activity_levels = measure_activity_levels[measure]
        fouts = [open(f"{output_dir}/{measure}_{metric}", "a", buffering = 1) for metric in ["accuracies", "auc", "pred"]]
        for level in activity_levels:
            print(f"{measure}={level}")
            for i in range(6):
                df_ds = list()
                for j in range(5):
                    df_path = f"{input_dir}/{measure}_dropout_{level}_{i * 5 + j}"
                    df_ds.append(pd.read_csv(df_path, dtype = {"lang": "string"}, sep = "\t", header = 0))
                df_ds = pd.concat(df_ds)
                selected_columns = list(df_ds.columns)[1:-1]
                for fout, model_metric in zip(fouts, train_model(df_ds, selected_columns)):
                    fout.write(f"{level}\t{i}\t{json.dumps(model_metric)}\n")
        for fout in fouts:
            fout.close()