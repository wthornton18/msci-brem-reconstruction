import tensorflow as tf
import scipy as sp
import numpy as np
import seaborn as sns
import pandas as pd
import psiK
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
    accuracy_score,
    plot_confusion_matrix,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
from skopt.space import Real, Integer
from skopt import gp_minimize
import warnings

warnings.filterwarnings("ignore")


class MLP_model:
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

    def model_(self):
        self.model = Sequential(
            [
                Dense(100, activation="tanh"),
                Dropout(0.2),
                Dense(100, activation="tanh"),
                Dense(2, activation="softmax"),
            ]
        )
        Adam = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model.compile(optimizer=Adam, loss=loss_fn, metrics=["accuracy"])
        return self.model

    def train(self, x_train, y_train, x_test, y_test, verbose=2):
        self_model = self.model_()
        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=verbose,
        )
        return history

    def eval(self, x_test, y_test):
        results = self.model.evaluate(x_test, y_test)
        # print("test loss, test acc:", results)
        return results

    def predict_proba(self, x_data):
        output = self.model.predict(x_data)
        return output

    def roc_curve_error(self, X_train, y_train, X_test, y_test, ratio=1):
        n_repeats = 10
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
        folds = [(train, val_temp) for train, val_temp in cv.split(X_train, y_train)]

        metrics = ["auc", "fpr", "tpr", "thresholds"]
        results = {
            "train": {m: [] for m in metrics},
            "val": {m: [] for m in metrics},
            "test": {m: [] for m in metrics},
        }

        dtest = (X_test, y_test)
        for train, test in tqdm(folds, total=len(folds)):
            dtrain = (X_train[train, :], y_train[train])
            dval = (X_train[test, :], y_train[test])

            model = self.model_().fit(dtrain[0], dtrain[1], verbose=0)
            sets = [dtrain, dval, dtest]
            for i, ds in enumerate(results.keys()):
                y_preds = model.predict_proba(sets[i][0])[:, 1]
                labels = sets[i][1]
                fpr, tpr, thresholds = roc_curve(labels, y_preds)
                results[ds]["fpr"].append(fpr)
                results[ds]["tpr"].append(tpr)
                results[ds]["thresholds"].append(thresholds)
                results[ds]["auc"].append(roc_auc_score(labels, y_preds))

        c_fill_train = "rgba(128, 252, 128, 0.2)"
        c_line_train = "rgba(128, 152, 128, 0.5)"
        c_line_main_train = "rgba(128, 0, 128, 1.0)"

        c_fill_val = "rgba(52, 152, 0, 0.2)"
        c_line_val = "rgba(52, 152, 0, 0.5)"
        c_line_main_val = "rgba(41, 128, 0, 1.0)"

        c_fill_test = "rgba(0, 152, 219, 0.2)"
        c_line_test = "rgba(0, 152, 219, 0.5)"
        c_line_main_test = "rgba(0, 128, 185, 1.0)"

        c_grid = "rgba(189, 195, 199, 0.5)"
        c_annot = "rgba(149, 165, 166, 0.5)"
        c_highlight = "rgba(192, 57, 43, 1.0)"
        fpr_mean = np.linspace(0, 1, 100)

        def tp_rates(kind, results):
            interp_tprs = []
            for i in range(n_repeats):
                fpr = results[kind]["fpr"][i]
                tpr = results[kind]["tpr"][i]
                interp_tpr = np.interp(fpr_mean, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tprs.append(interp_tpr)
            tpr_mean = np.mean(interp_tprs, axis=0)
            tpr_mean[-1] = 1.0
            tpr_std = 2 * np.std(interp_tprs, axis=0)
            tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
            tpr_lower = tpr_mean - tpr_std
            auc = np.mean(results[kind]["auc"])
            return tpr_upper, tpr_mean, tpr_lower, auc

        kind = "val"
        try:
            title = "ROC Curve with signal to background ratio of: {:.2f}".format(ratio)
        except:
            title = "ROC Curve"
        train_tpr_upper, train_tpr_mean, train_tpr_lower, train_auc = tp_rates(
            "train", results
        )
        val_tpr_upper, val_tpr_mean, val_tpr_lower, val_auc = tp_rates("val", results)
        test_tpr_upper, test_tpr_mean, test_tpr_lower, test_auc = tp_rates(
            "test", results
        )
        fig = go.Figure(
            [
                go.Scatter(
                    x=fpr_mean,
                    y=train_tpr_upper,
                    line=dict(color=c_line_train, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="upper",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=train_tpr_lower,
                    fill="tonexty",
                    fillcolor=c_fill_train,
                    line=dict(color=c_line_train, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="lower",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=train_tpr_mean,
                    line=dict(color=c_line_main_train, width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"Train_AUC: {train_auc:.3f}",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=val_tpr_upper,
                    line=dict(color=c_line_val, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="upper",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=val_tpr_lower,
                    fill="tonexty",
                    fillcolor=c_fill_val,
                    line=dict(color=c_line_val, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="lower",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=val_tpr_mean,
                    line=dict(color=c_line_main_val, width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"Val_AUC: {val_auc:.3f}",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=test_tpr_upper,
                    line=dict(color=c_line_test, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="upper",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=test_tpr_lower,
                    fill="tonexty",
                    fillcolor=c_fill_test,
                    line=dict(color=c_line_test, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="lower",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=test_tpr_mean,
                    line=dict(color=c_line_main_test, width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"Test_AUC: {test_auc:.3f}",
                ),
            ]
        )
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
        fig.update_layout(
            title="ROC Curve for MLP model",
            template="plotly_white",
            title_x=0.5,
            xaxis_title="1 - Specificity",
            yaxis_title="Sensitivity",
            width=800,
            height=800,
            legend=dict(
                yanchor="bottom", xanchor="right", x=0.95, y=0.01, font=dict(size=24)
            ),
            yaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)),
            xaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)),
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=50,  # top margin
            ),
        )

        fig.update_yaxes(
            range=[0, 1],
            gridcolor=c_grid,
            scaleanchor="x",
            scaleratio=1,
            linecolor="black",
        )
        fig.update_xaxes(
            range=[0, 1], gridcolor=c_grid, constrain="domain", linecolor="black"
        )
        # import os

        # if not os.path.exists("images"):
        #     os.mkdir("images")

        # fig.write_image("images/viva/roc_curve_mlp.svg")
        fig.show()
        return results, model

    def plot_loss_acc(self, history):
        fig = plt.figure(figsize=(20, 10))

        fig.add_subplot(121)
        plt.plot(history.history["loss"], label="Train")
        plt.plot(history.history["val_loss"], label="Validation")
        plt.xlabel("Epoch", fontsize=18)
        plt.ylabel("Categorical cross entropy loss", fontsize=18)
        plt.title("Loss vs epoch", fontsize=18)
        plt.legend(fontsize=18)

        fig.add_subplot(122)
        plt.plot(history.history["accuracy"], label="Train")
        plt.plot(history.history["val_accuracy"], label="Validation")
        plt.xlabel("Epoch", fontsize=18)
        plt.ylabel("Accuracy", fontsize=18)
        plt.title("Accuracy vs epoch", fontsize=18)
        plt.legend(fontsize=18)
        plt.show()

    def plot_roc_curve(
        self, training_data, training_labels, validation_data, validation_labels
    ):
        y_train_pred = self.predict_proba(training_data)[:, 1]
        y_val_pred = self.predict_proba(validation_data)[:, 1]

        nn_fpr_keras_val, nn_tpr_keras_val, nn_thresholds_keras_val = roc_curve(
            validation_labels, y_val_pred
        )
        auc_keras_val = auc(nn_fpr_keras_val, nn_tpr_keras_val)

        nn_fpr_keras_train, nn_tpr_keras_train, nn_thresholds_keras_train = roc_curve(
            training_labels, y_train_pred
        )
        auc_keras_train = auc(nn_fpr_keras_train, nn_tpr_keras_train)
        plt.figure(figsize=(15, 10))
        plt.plot(
            nn_fpr_keras_train,
            nn_tpr_keras_train,
            marker=".",
            label="Neural Network train (auc = %0.3f)" % auc_keras_train,
        )
        plt.plot(
            nn_fpr_keras_val,
            nn_tpr_keras_val,
            marker=".",
            label="Neural Network val (auc = %0.3f)" % auc_keras_val,
        )
        plt.xlabel("fpr", fontsize=18)
        plt.ylabel("tpr", fontsize=18)
        plt.legend(fontsize=18)
        plt.show()

    def conf_mat_plot(
        self, training_data, training_labels, validation_data, validation_labels
    ):
        y_train_pred = self.predict_proba(training_data)
        pos_class_train, neg_class_train = y_train_pred[:, 1], y_train_pred[:, 0]
        cm_train = confusion_matrix(training_labels, pos_class_train.round())

        y_test_pred = self.predict_proba(validation_data)
        pos_class_test, neg_class_test = y_test_pred[:, 1], y_test_pred[:, 0]
        cm_test = confusion_matrix(validation_labels, pos_class_test.round())

        fig, axs = plt.subplots(2, 1, figsize=(18, 18))
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_train)
        disp1.plot(cmap=plt.cm.Blues, ax=axs[0])
        axs[0].set_title("Training Data on MLP")
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_test)
        disp2.plot(cmap=plt.cm.Blues, ax=axs[1])
        axs[1].set_title("Validation Data on MLP")
        plt.show()

    def decision_function(
        self, training_data, training_labels, validation_data, validation_labels
    ):
        ind1_val = list(np.where(validation_labels == 1))[0]
        ind0_val = list(np.where(validation_labels == 0))[0]
        y_pred1_val = self.predict_proba(validation_data[ind1_val])[:, 1]
        y_pred0_val = self.predict_proba(validation_data[ind0_val])[:, 1]

        ind1_train = list(np.where(training_labels == 1))[0]
        ind0_train = list(np.where(training_labels == 0))[0]
        y_pred1_train = self.predict_proba(training_data[ind1_train])[:, 1]
        y_pred0_train = self.predict_proba(training_data[ind0_train])[:, 1]

        width = 3
        fig = plt.figure(figsize=(15, 10))
        fig.add_subplot(121)
        sns.distplot(
            y_pred1_val,
            bins=100,
            color="blue",
            norm_hist=True,
            hist=False,
            kde_kws={"shade": True, "linewidth": width},
            label="Positive class",
        )
        sns.distplot(
            y_pred0_val,
            bins=100,
            color="green",
            norm_hist=True,
            hist=False,
            kde_kws={"shade": True, "linewidth": width},
            label="Negative class",
        )
        plt.xlim(0, 1)
        plt.legend(loc="upper center", fontsize=18)

        fig.add_subplot(122)
        sns.distplot(
            y_pred1_train,
            bins=100,
            color="blue",
            norm_hist=True,
            hist=False,
            kde_kws={"shade": True, "linewidth": width},
            label="Positive class",
        )
        sns.distplot(
            y_pred0_train,
            bins=100,
            color="green",
            norm_hist=True,
            hist=False,
            kde_kws={"shade": True, "linewidth": width},
            label="Negative class",
        )
        plt.xlim(0, 1)
        plt.legend(loc="upper center", fontsize=18)
        plt.show()

    def time_per_photon(self, X, iter: int = 10):
        model = self.model
        st = time.time()
        for _ in range(iter):
            y = model.predict(X)
        en = time.time()
        full_time = (en - st) / (iter * len(X))
        return full_time


class SVM_model:
    def __init__(
        self, kernel: str = "rbf", hardness: float = 1, gamma="scale", default=True
    ):
        self.kernel = kernel
        self.hardness = hardness
        self.gamma = gamma
        self.default = default
        if self.default:
            self.params = {
                "kernel": "rbf",
                "C": 1,
                "gamma": "scale",
                "probability": True,
            }
        else:
            self.params = {
                "kernel": self.kernel,
                "C": self.hardness,
                "gamma": self.gamma,
                "probability": True,
            }

    def model_(self):
        self.model = SVC().set_params(**self.params)
        return self.model

    def train(self, x_train, y_train):
        self.model = self.model_()
        self.model.fit(x_train, y_train)

    def predict_proba(self, x_data):
        output = self.model.predict_proba(x_data)
        return output

    def predict(self, x_data):
        output = self.model.predict(x_data)
        return output

    def eval(self, x_data, y_data):
        pred = self.predict(x_data)
        return np.mean(pred == y_data)

    def plot_roc_curve(
        self, training_data, training_labels, validation_data, validation_labels
    ):
        y_train_pred = self.predict_proba(training_data)[:, 1]
        y_val_pred = self.predict_proba(validation_data)[:, 1]

        nn_fpr_keras_val, nn_tpr_keras_val, nn_thresholds_keras_val = roc_curve(
            validation_labels, y_val_pred
        )
        auc_keras_val = auc(nn_fpr_keras_val, nn_tpr_keras_val)

        nn_fpr_keras_train, nn_tpr_keras_train, nn_thresholds_keras_train = roc_curve(
            training_labels, y_train_pred
        )
        auc_keras_train = auc(nn_fpr_keras_train, nn_tpr_keras_train)
        plt.figure(figsize=(15, 10))
        plt.plot(
            nn_fpr_keras_train,
            nn_tpr_keras_train,
            marker=".",
            label="SVM train (auc = %0.3f)" % auc_keras_train,
        )
        plt.plot(
            nn_fpr_keras_val,
            nn_tpr_keras_val,
            marker=".",
            label="SVM val (auc = %0.3f)" % auc_keras_val,
        )
        plt.xlabel("fpr", fontsize=18)
        plt.ylabel("tpr", fontsize=18)
        plt.legend(fontsize=18)
        plt.show()

    def roc_curve_error(self, X_train, y_train, X_test, y_test, ratio=1):
        n_repeats = 10
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
        folds = [(train, val_temp) for train, val_temp in cv.split(X_train, y_train)]

        metrics = ["auc", "fpr", "tpr", "thresholds"]
        results = {
            "train": {m: [] for m in metrics},
            "val": {m: [] for m in metrics},
            "test": {m: [] for m in metrics},
        }

        dtest = (X_test, y_test)
        for train, test in tqdm(folds, total=len(folds)):
            dtrain = (X_train[train, :], y_train[train])
            dval = (X_train[test, :], y_train[test])

            model = self.model_().fit(dtrain[0], dtrain[1])
            sets = [dtrain, dval, dtest]
            for i, ds in enumerate(results.keys()):
                y_preds = model.predict_proba(sets[i][0])[:, 1]
                labels = sets[i][1]
                fpr, tpr, thresholds = roc_curve(labels, y_preds)
                results[ds]["fpr"].append(fpr)
                results[ds]["tpr"].append(tpr)
                results[ds]["thresholds"].append(thresholds)
                results[ds]["auc"].append(roc_auc_score(labels, y_preds))

        c_fill_train = "rgba(128, 252, 128, 0.2)"
        c_line_train = "rgba(128, 152, 128, 0.5)"
        c_line_main_train = "rgba(128, 0, 128, 1.0)"

        c_fill_val = "rgba(52, 152, 0, 0.2)"
        c_line_val = "rgba(52, 152, 0, 0.5)"
        c_line_main_val = "rgba(41, 128, 0, 1.0)"

        c_fill_test = "rgba(0, 152, 219, 0.2)"
        c_line_test = "rgba(0, 152, 219, 0.5)"
        c_line_main_test = "rgba(0, 128, 185, 1.0)"

        c_grid = "rgba(189, 195, 199, 0.5)"
        c_annot = "rgba(149, 165, 166, 0.5)"
        c_highlight = "rgba(192, 57, 43, 1.0)"
        fpr_mean = np.linspace(0, 1, 100)

        def tp_rates(kind, results):
            interp_tprs = []
            for i in range(n_repeats):
                fpr = results[kind]["fpr"][i]
                tpr = results[kind]["tpr"][i]
                interp_tpr = np.interp(fpr_mean, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tprs.append(interp_tpr)
            tpr_mean = np.mean(interp_tprs, axis=0)
            tpr_mean[-1] = 1.0
            tpr_std = 2 * np.std(interp_tprs, axis=0)
            tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
            tpr_lower = tpr_mean - tpr_std
            auc = np.mean(results[kind]["auc"])
            return tpr_upper, tpr_mean, tpr_lower, auc

        kind = "val"
        try:
            title = "ROC Curve with signal to background ratio of: {:.2f}".format(ratio)
        except:
            title = "ROC Curve"
        train_tpr_upper, train_tpr_mean, train_tpr_lower, train_auc = tp_rates(
            "train", results
        )
        val_tpr_upper, val_tpr_mean, val_tpr_lower, val_auc = tp_rates("val", results)
        test_tpr_upper, test_tpr_mean, test_tpr_lower, test_auc = tp_rates(
            "test", results
        )
        fig = go.Figure(
            [
                go.Scatter(
                    x=fpr_mean,
                    y=train_tpr_upper,
                    line=dict(color=c_line_train, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="upper",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=train_tpr_lower,
                    fill="tonexty",
                    fillcolor=c_fill_train,
                    line=dict(color=c_line_train, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="lower",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=train_tpr_mean,
                    line=dict(color=c_line_main_train, width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"Train_AUC: {train_auc:.3f}",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=val_tpr_upper,
                    line=dict(color=c_line_val, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="upper",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=val_tpr_lower,
                    fill="tonexty",
                    fillcolor=c_fill_val,
                    line=dict(color=c_line_val, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="lower",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=val_tpr_mean,
                    line=dict(color=c_line_main_val, width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"Val_AUC: {val_auc:.3f}",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=test_tpr_upper,
                    line=dict(color=c_line_test, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="upper",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=test_tpr_lower,
                    fill="tonexty",
                    fillcolor=c_fill_test,
                    line=dict(color=c_line_test, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="lower",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=test_tpr_mean,
                    line=dict(color=c_line_main_test, width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"Test_AUC: {test_auc:.3f}",
                ),
            ]
        )
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
        fig.update_layout(
            title="ROC Curve for SVM model",
            template="plotly_white",
            title_x=0.5,
            xaxis_title="1 - Specificity",
            yaxis_title="Sensitivity",
            width=800,
            height=800,
            legend=dict(
                yanchor="bottom", xanchor="right", x=0.95, y=0.01, font=dict(size=24)
            ),
            yaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)),
            xaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)),
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=50,  # top margin
            ),
        )

        fig.update_yaxes(
            range=[0, 1],
            gridcolor=c_grid,
            scaleanchor="x",
            scaleratio=1,
            linecolor="black",
        )
        fig.update_xaxes(
            range=[0, 1], gridcolor=c_grid, constrain="domain", linecolor="black"
        )
        # import os

        # if not os.path.exists("images"):
        #     os.mkdir("images")

        # fig.write_image("images/viva/roc_curve_svm2.svg")
        fig.show()
        return results, model

    def conf_mat_plot(
        self, training_data, training_labels, validation_data, validation_labels
    ):
        y_train_pred = self.predict_proba(training_data)
        pos_class_train, neg_class_train = y_train_pred[:, 1], y_train_pred[:, 0]
        cm_train = confusion_matrix(training_labels, pos_class_train.round())

        y_test_pred = self.predict_proba(validation_data)
        pos_class_test, neg_class_test = y_test_pred[:, 1], y_test_pred[:, 0]
        cm_test = confusion_matrix(validation_labels, pos_class_test.round())

        fig, axs = plt.subplots(2, 1, figsize=(18, 18))
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_train)
        disp1.plot(cmap=plt.cm.Blues, ax=axs[0])
        axs[0].set_title("Training Data on SVM")
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_test)
        disp2.plot(cmap=plt.cm.Blues, ax=axs[1])
        axs[1].set_title("Validation Data on SVM")
        plt.show()

    def decision_function(
        self, training_data, training_labels, validation_data, validation_labels
    ):
        ind1_val = list(np.where(validation_labels == 1))[0]
        ind0_val = list(np.where(validation_labels == 0))[0]
        y_pred1_val = self.predict_proba(validation_data[ind1_val])[:, 1]
        y_pred0_val = self.predict_proba(validation_data[ind0_val])[:, 1]

        ind1_train = list(np.where(training_labels == 1))[0]
        ind0_train = list(np.where(training_labels == 0))[0]
        y_pred1_train = self.predict_proba(training_data[ind1_train])[:, 1]
        y_pred0_train = self.predict_proba(training_data[ind0_train])[:, 1]

        width = 3
        fig = plt.figure(figsize=(15, 10))
        fig.add_subplot(121)
        sns.distplot(
            y_pred1_val,
            bins=100,
            color="blue",
            norm_hist=True,
            hist=False,
            kde_kws={"shade": True, "linewidth": width},
            label="Positive class",
        )
        sns.distplot(
            y_pred0_val,
            bins=100,
            color="green",
            norm_hist=True,
            hist=False,
            kde_kws={"shade": True, "linewidth": width},
            label="Negative class",
        )
        plt.xlim(0, 1)
        plt.legend(loc="upper center", fontsize=18)

        fig.add_subplot(122)
        sns.distplot(
            y_pred1_train,
            bins=100,
            color="blue",
            norm_hist=True,
            hist=False,
            kde_kws={"shade": True, "linewidth": width},
            label="Positive class",
        )
        sns.distplot(
            y_pred0_train,
            bins=100,
            color="green",
            norm_hist=True,
            hist=False,
            kde_kws={"shade": True, "linewidth": width},
            label="Negative class",
        )
        plt.xlim(0, 1)
        plt.legend(loc="upper center", fontsize=18)
        plt.show()

    def time_per_photon(self, X, iter: int = 10):
        model = self.model
        st = time.time()
        for _ in range(iter):
            y = model.predict(X)
        en = time.time()
        full_time = (en - st) / (iter * len(X))
        return full_time


class XGB_model:
    def __init__(self, default: bool = True):
        self.default = default

        if self.default:
            self.params = {
                "early_stopping_rounds": 20,
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "silent": True,
                "verbosity": 0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "booster": "gbtree",
                "n_jobs": -1,
                "nthread": None,
                "gamma": 0,
                "min_child_weight": 1,
                "max_delta_step": 0,
                "subsample": 1,
                "colsample_bytree": 1,
                "colsample_bylevel": 1,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "scale_pos_weight": 1,
                "base_score": 0.5,
                "use_label_encoder": False,
                "random_state": 42,
                "seed": 42,
            }

        else:
            self.params = {
                "early_stopping_rounds": 20,
                "max_depth": 9,
                "learning_rate": 0.1,
                "n_estimators": 1000,
                "silent": True,
                "verbosity": 0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "booster": "gbtree",
                "n_jobs": -1,
                "nthread": None,
                "gamma": 0,
                "min_child_weight": 1,
                "max_delta_step": 0,
                "subsample": 1,
                "colsample_bytree": 1,
                "colsample_bylevel": 1,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "scale_pos_weight": 1,
                "base_score": 0.5,
                "use_label_encoder": False,
                "random_state": 42,
                "seed": 42,
            }

    def model_(self):
        self.model = XGBClassifier(use_label_encoder=False).set_params(**self.params)
        return self.model

    def train(self, x_train, y_train):
        self.model = self.model_()
        self.model.fit(x_train, y_train).set_params(**self.params)

    def predict_proba(self, x_data):
        output = self.model.predict_proba(x_data)
        return output

    def predict(self, x_data):
        output = self.model.predict(x_data)
        return output

    def eval(self, x_data, y_data):
        pred = self.predict(x_data)
        return np.mean(pred == y_data)

    def plot_roc_curve(
        self, training_data, training_labels, validation_data, validation_labels
    ):
        y_train_pred = self.predict_proba(training_data)[:, 1]
        y_val_pred = self.predict_proba(validation_data)[:, 1]

        nn_fpr_keras_val, nn_tpr_keras_val, nn_thresholds_keras_val = roc_curve(
            validation_labels, y_val_pred
        )
        auc_keras_val = auc(nn_fpr_keras_val, nn_tpr_keras_val)

        nn_fpr_keras_train, nn_tpr_keras_train, nn_thresholds_keras_train = roc_curve(
            training_labels, y_train_pred
        )
        auc_keras_train = auc(nn_fpr_keras_train, nn_tpr_keras_train)
        plt.figure(figsize=(15, 10))
        plt.plot(
            nn_fpr_keras_train,
            nn_tpr_keras_train,
            marker=".",
            label="SVM train (auc = %0.3f)" % auc_keras_train,
        )
        plt.plot(
            nn_fpr_keras_val,
            nn_tpr_keras_val,
            marker=".",
            label="SVM val (auc = %0.3f)" % auc_keras_val,
        )
        plt.xlabel("fpr", fontsize=18)
        plt.ylabel("tpr", fontsize=18)
        plt.legend(fontsize=18)
        plt.show()

    def plot_loss_valid(self, X_train, y_train, X_test, y_test):
        # fit model no training data
        model = self.model()
        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(
            X_train,
            y_train,
            eval_metric=["error", "logloss"],
            eval_set=eval_set,
            verbose=False,
        )

        # make predictions for test data
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]

        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        # retrieve performance metrics
        results = model.evals_result()
        epochs = len(results["validation_0"]["error"])
        x_axis = range(0, epochs)

        # plot log loss
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(x_axis, results["validation_0"]["logloss"], label="Train")
        ax.plot(x_axis, results["validation_1"]["logloss"], label="Test")
        ax.legend()

        plt.ylabel("Log Loss")
        plt.title("XGBoost Log Loss")
        plt.show()

        # plot classification error
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(x_axis, results["validation_0"]["error"], label="Train")
        ax.plot(x_axis, results["validation_1"]["error"], label="Test")
        ax.legend()

        plt.ylabel("Classification Error")
        plt.title("XGBoost Classification Error")
        plt.show()

    def roc_curve_error(self, X_train, y_train, X_test, y_test, default=False, ratio=1):
        n_repeats = 10
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=101)
        folds = [(train, val_temp) for train, val_temp in cv.split(X_train, y_train)]

        metrics = ["auc", "fpr", "tpr", "thresholds"]
        results = {
            "train": {m: [] for m in metrics},
            "val": {m: [] for m in metrics},
            "test": {m: [] for m in metrics},
        }
        dtest = (X_test, y_test)
        for train, test in tqdm(folds, total=len(folds)):
            dtrain = (X_train[train, :], y_train[train])
            dval = (X_train[test, :], y_train[test])

            model = self.model_().fit(dtrain[0], dtrain[1], eval_set=[dval], verbose=0)
            sets = [dtrain, dval, dtest]
            for i, ds in enumerate(results.keys()):
                y_preds = model.predict_proba(sets[i][0])[:, 1]
                labels = sets[i][1]
                fpr, tpr, thresholds = roc_curve(labels, y_preds)
                results[ds]["fpr"].append(fpr)
                results[ds]["tpr"].append(tpr)
                results[ds]["thresholds"].append(thresholds)
                results[ds]["auc"].append(roc_auc_score(labels, y_preds))

        c_fill_train = "rgba(128, 252, 128, 0.2)"
        c_line_train = "rgba(128, 152, 128, 0.5)"
        c_line_main_train = "rgba(128, 0, 128, 1.0)"

        c_fill_val = "rgba(52, 152, 0, 0.2)"
        c_line_val = "rgba(52, 152, 0, 0.5)"
        c_line_main_val = "rgba(41, 128, 0, 1.0)"

        c_fill_test = "rgba(0, 152, 219, 0.2)"
        c_line_test = "rgba(0, 152, 219, 0.5)"
        c_line_main_test = "rgba(0, 128, 185, 1.0)"

        c_grid = "rgba(189, 195, 199, 0.5)"
        c_annot = "rgba(149, 165, 166, 0.5)"
        c_highlight = "rgba(192, 57, 43, 1.0)"
        fpr_mean = np.linspace(0, 1, 100)

        def tp_rates(kind, results):
            interp_tprs = []
            for i in range(n_repeats):
                fpr = results[kind]["fpr"][i]
                tpr = results[kind]["tpr"][i]
                interp_tpr = np.interp(fpr_mean, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tprs.append(interp_tpr)
            tpr_mean = np.mean(interp_tprs, axis=0)
            tpr_mean[-1] = 1.0
            tpr_std = 2 * np.std(interp_tprs, axis=0)
            tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
            tpr_lower = tpr_mean - tpr_std
            auc = np.mean(results[kind]["auc"])
            return tpr_upper, tpr_mean, tpr_lower, auc

        kind = "val"
        try:
            title = "ROC Curve with signal to background ratio of: {:.2f}".format(ratio)
        except:
            title = "ROC Curve"
        train_tpr_upper, train_tpr_mean, train_tpr_lower, train_auc = tp_rates(
            "train", results
        )
        val_tpr_upper, val_tpr_mean, val_tpr_lower, val_auc = tp_rates("val", results)
        test_tpr_upper, test_tpr_mean, test_tpr_lower, test_auc = tp_rates(
            "test", results
        )
        fig = go.Figure(
            [
                go.Scatter(
                    x=fpr_mean,
                    y=train_tpr_upper,
                    line=dict(color=c_line_train, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="upper",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=train_tpr_lower,
                    fill="tonexty",
                    fillcolor=c_fill_train,
                    line=dict(color=c_line_train, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="lower",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=train_tpr_mean,
                    line=dict(color=c_line_main_train, width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"Train_AUC: {train_auc:.3f}",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=val_tpr_upper,
                    line=dict(color=c_line_val, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="upper",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=val_tpr_lower,
                    fill="tonexty",
                    fillcolor=c_fill_val,
                    line=dict(color=c_line_val, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="lower",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=val_tpr_mean,
                    line=dict(color=c_line_main_val, width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"Val_AUC: {val_auc:.3f}",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=test_tpr_upper,
                    line=dict(color=c_line_test, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="upper",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=test_tpr_lower,
                    fill="tonexty",
                    fillcolor=c_fill_test,
                    line=dict(color=c_line_test, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="lower",
                ),
                go.Scatter(
                    x=fpr_mean,
                    y=test_tpr_mean,
                    line=dict(color=c_line_main_test, width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"Test_AUC: {test_auc:.3f}",
                ),
            ]
        )
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
        fig.update_layout(
            title="ROC Curve for XGB model",
            template="plotly_white",
            title_x=0.5,
            xaxis_title="1 - Specificity",
            yaxis_title="Sensitivity",
            width=800,
            height=800,
            legend=dict(
                yanchor="bottom", xanchor="right", x=0.95, y=0.01, font=dict(size=24)
            ),
            yaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)),
            xaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)),
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=50,  # top margin
            ),
        )

        fig.update_yaxes(
            range=[0, 1],
            gridcolor=c_grid,
            scaleanchor="x",
            scaleratio=1,
            linecolor="black",
        )
        fig.update_xaxes(
            range=[0, 1], gridcolor=c_grid, constrain="domain", linecolor="black"
        )
        # import os

        # if not os.path.exists("images"):
        #     os.mkdir("images")

        # fig.write_image("images/viva/roc_curve_xgb2.svg")
        fig.show()
        return results, model

    def precision_recall(
        self, X_train, y_train, X_test, y_test, default=False, ratio=1
    ):
        n_repeats = 10
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=101)
        folds = [(train, val_temp) for train, val_temp in cv.split(X_train, y_train)]

        metrics = ["ap", "precision", "recall", "thresholds"]
        results = {
            "train": {m: [] for m in metrics},
            "val": {m: [] for m in metrics},
            "test": {m: [] for m in metrics},
        }
        if default:
            params = {
                "early_stopping_rounds": 20,
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "silent": True,
                "verbosity": 0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "booster": "gbtree",
                "n_jobs": -1,
                "nthread": None,
                "gamma": 0,
                "min_child_weight": 1,
                "max_delta_step": 0,
                "subsample": 1,
                "colsample_bytree": 1,
                "colsample_bylevel": 1,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "scale_pos_weight": 1,
                "base_score": 0.5,
                "use_label_encoder": False,
                "random_state": 42,
                "seed": 42,
            }

        else:
            params = {
                "early_stopping_rounds": 20,
                "max_depth": 9,
                "learning_rate": 0.1,
                "n_estimators": 1000,
                "silent": True,
                "verbosity": 0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "booster": "gbtree",
                "n_jobs": -1,
                "nthread": None,
                "gamma": 0,
                "min_child_weight": 1,
                "max_delta_step": 0,
                "subsample": 1,
                "colsample_bytree": 1,
                "colsample_bylevel": 1,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "scale_pos_weight": 1,
                "base_score": 0.5,
                "use_label_encoder": False,
                "random_state": 42,
                "seed": 42,
            }

        dtest = (X_test, y_test)
        for train, test in tqdm(folds, total=len(folds)):
            dtrain = (X_train.iloc[train, :], y_train.iloc[train])
            dval = (X_train.iloc[test, :], y_train.iloc[test])

            model = (
                self.model_()
                .set_params(**params)
                .fit(dtrain[0], dtrain[1], eval_set=[dval], verbose=0)
            )
            sets = [dtrain, dval, dtest]
            for i, ds in enumerate(results.keys()):
                y_preds = model.predict_proba(sets[i][0])[:, 1]
                labels = sets[i][1]
                precision, recall, thresholds = precision_recall_curve(labels, y_preds)
                results[ds]["precision"].append(precision)
                results[ds]["recall"].append(recall)
                results[ds]["thresholds"].append(thresholds)
                results[ds]["ap"].append(average_precision_score(labels, y_preds))

        c_fill_train = "rgba(128, 252, 128, 0.2)"
        c_line_train = "rgba(128, 152, 128, 0.5)"
        c_line_main_train = "rgba(128, 0, 128, 1.0)"

        c_fill_val = "rgba(52, 152, 0, 0.2)"
        c_line_val = "rgba(52, 152, 0, 0.5)"
        c_line_main_val = "rgba(41, 128, 0, 1.0)"

        c_fill_test = "rgba(0, 152, 219, 0.2)"
        c_line_test = "rgba(0, 152, 219, 0.5)"
        c_line_main_test = "rgba(0, 128, 185, 1.0)"

        c_grid = "rgba(189, 195, 199, 0.5)"
        c_annot = "rgba(149, 165, 166, 0.5)"
        c_highlight = "rgba(192, 57, 43, 1.0)"
        recall_mean = np.linspace(0, 1, 1000)

        def tp_rates(kind, results):
            interp_precisions = []
            for i in range(n_repeats):
                recall = np.array(results[kind]["recall"][i])
                precision = np.array(results[kind]["precision"][i])
                inds = recall.argsort()
                recall_s = recall[inds]
                precision_s = precision[inds]
                interp_precision = np.interp(recall_mean, recall_s, precision_s)
                # interp_precision[0] = 0.0
                interp_precisions.append(interp_precision)
            precision_mean = np.mean(interp_precisions, axis=0)
            # precision_mean[-1] = 1.0
            precision_std = 2 * np.std(interp_precisions, axis=0)
            precision_upper = np.clip(precision_mean + precision_std, 0, 1)
            precision_lower = precision_mean - precision_std
            ap = np.mean(results[kind]["ap"])
            return precision_upper, precision_mean, precision_lower, ap

        kind = "val"
        try:
            title = "Precision-Recall Curve with signal to background ratio of: {:.2f}".format(
                ratio
            )
        except:
            title = "Precision-Recall Curve"

        train_tpr_upper, train_tpr_mean, train_tpr_lower, train_auc = tp_rates(
            "train", results
        )
        val_tpr_upper, val_tpr_mean, val_tpr_lower, val_auc = tp_rates("val", results)
        test_tpr_upper, test_tpr_mean, test_tpr_lower, test_auc = tp_rates(
            "test", results
        )
        fig = go.Figure(
            [
                go.Scatter(
                    x=recall_mean,
                    y=train_tpr_upper,
                    line=dict(color=c_line_train, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="upper",
                ),
                go.Scatter(
                    x=recall_mean,
                    y=train_tpr_lower,
                    fill="tonexty",
                    fillcolor=c_fill_train,
                    line=dict(color=c_line_train, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="lower",
                ),
                go.Scatter(
                    x=recall_mean,
                    y=train_tpr_mean,
                    line=dict(color=c_line_main_train, width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"Train_AP: {train_auc:.3f}",
                ),
                go.Scatter(
                    x=recall_mean,
                    y=val_tpr_upper,
                    line=dict(color=c_line_val, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="upper",
                ),
                go.Scatter(
                    x=recall_mean,
                    y=val_tpr_lower,
                    fill="tonexty",
                    fillcolor=c_fill_val,
                    line=dict(color=c_line_val, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="lower",
                ),
                go.Scatter(
                    x=recall_mean,
                    y=val_tpr_mean,
                    line=dict(color=c_line_main_val, width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"Val_AP: {val_auc:.3f}",
                ),
                go.Scatter(
                    x=recall_mean,
                    y=test_tpr_upper,
                    line=dict(color=c_line_test, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="upper",
                ),
                go.Scatter(
                    x=recall_mean,
                    y=test_tpr_lower,
                    fill="tonexty",
                    fillcolor=c_fill_test,
                    line=dict(color=c_line_test, width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="lower",
                ),
                go.Scatter(
                    x=recall_mean,
                    y=test_tpr_mean,
                    line=dict(color=c_line_main_test, width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"Test_AP: {test_auc:.3f}",
                ),
            ]
        )
        # fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
        fig.update_layout(
            # title=title,
            template="plotly_white",
            title_x=0.5,
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=800,
            height=800,
            legend=dict(
                yanchor="bottom", xanchor="right", x=0.95, y=0.01, font=dict(size=24)
            ),
            yaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)),
            xaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)),
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=0,  # top margin
            ),
        )

        fig.update_yaxes(
            range=[0, 1],
            gridcolor=c_grid,
            scaleanchor="x",
            scaleratio=1,
            linecolor="black",
        )
        fig.update_xaxes(
            range=[0, 1], gridcolor=c_grid, constrain="domain", linecolor="black"
        )
        # import os

        # if not os.path.exists("images"):
        #     os.mkdir("images")

        # fig.write_image("images/Precision_recall_test.svg")
        # fig.show()
        return results, model

    def optimise_model(self, model, X_train, y_train, X_test, y_test):

        # defining the space
        search_space = [
            Real(0.5, 1.0, name="colsample_bylevel"),
            Real(0.5, 1.0, name="colsample_bytree"),
            Real(0.0, 1.0, name="gamma"),
            Real(0.0001, 0.01, name="learning_rate"),
            Real(0.1, 10, name="max_delta_step"),
            Integer(3, 15, name="max_depth"),
            Real(1, 50, name="min_child_weight"),
            Integer(10, 1500, name="n_estimators"),
            Real(0.1, 100, name="reg_alpha"),
            Real(0.1, 100, name="reg_lambda"),
            Real(0.5, 1.0, name="subsample"),
        ]

        # collecting the fitted models and model performance
        models = []
        train_scores = []
        test_scores = []
        curr_model_hyper_params = [
            "colsample_bylevel",
            "colsample_bytree",
            "gamma",
            "learning_rate",
            "max_delta_step",
            "max_depth",
            "min_child_weight",
            "n_estimators",
            "reg_alpha",
            "reg_lambda",
            "subsample",
        ]

        # function to fit the model and return the performance of the model
        def return_model_assessment(args, X_train, y_train, X_test, y_test):
            # global models, train_scores, test_scores, curr_model_hyper_params
            params = {
                curr_model_hyper_params[i]: args[i]
                for i, j in enumerate(curr_model_hyper_params)
            }
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                early_stopping_rounds=20,
            )
            model.set_params(**params)
            fitted_model = model.fit(X_train, y_train, sample_weight=None)
            models.append(fitted_model)
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            train_score = f1_score(train_predictions, y_train)
            test_score = f1_score(test_predictions, y_test)
            train_scores.append(train_score)
            test_scores.append(test_score)
            return 1 - test_score

        objective_function = partial(
            return_model_assessment,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # running the algorithm
        n_calls = 2000  # number of times you want to train your model
        results = gp_minimize(
            objective_function,
            search_space,
            base_estimator=None,
            n_calls=n_calls,
            n_random_starts=n_calls - 1,
            random_state=42,
            verbose=True,
            n_jobs=-1,
        )

        return results, models, train_scores, test_scores

    def conf_mat_plot(
        self, training_data, training_labels, validation_data, validation_labels
    ):
        y_train_pred = self.predict_proba(training_data)
        pos_class_train, neg_class_train = y_train_pred[:, 1], y_train_pred[:, 0]
        cm_train = confusion_matrix(training_labels, pos_class_train.round())

        y_test_pred = self.predict_proba(validation_data)
        pos_class_test, neg_class_test = y_test_pred[:, 1], y_test_pred[:, 0]
        cm_test = confusion_matrix(validation_labels, pos_class_test.round())

        fig, axs = plt.subplots(2, 1, figsize=(18, 18))
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_train)
        disp1.plot(cmap=plt.cm.Blues, ax=axs[0])
        axs[0].set_title("Training Data on XGBoost")
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_test)
        disp2.plot(cmap=plt.cm.Blues, ax=axs[1])
        axs[1].set_title("Validation Data on XGBoost")
        plt.show()

    def decision_function(
        self, training_data, training_labels, validation_data, validation_labels
    ):
        ind1_val = list(np.where(validation_labels == 1))[0]
        ind0_val = list(np.where(validation_labels == 0))[0]
        y_pred1_val = self.predict_proba(validation_data[ind1_val])[:, 1]
        y_pred0_val = self.predict_proba(validation_data[ind0_val])[:, 1]

        ind1_train = list(np.where(training_labels == 1))[0]
        ind0_train = list(np.where(training_labels == 0))[0]
        y_pred1_train = self.predict_proba(training_data[ind1_train])[:, 1]
        y_pred0_train = self.predict_proba(training_data[ind0_train])[:, 1]

        width = 3
        fig = plt.figure(figsize=(15, 10))
        fig.add_subplot(121)
        sns.distplot(
            y_pred1_val,
            bins=100,
            color="blue",
            norm_hist=True,
            hist=False,
            kde_kws={"shade": True, "linewidth": width},
            label="Positive class",
        )
        sns.distplot(
            y_pred0_val,
            bins=100,
            color="green",
            norm_hist=True,
            hist=False,
            kde_kws={"shade": True, "linewidth": width},
            label="Negative class",
        )
        plt.xlim(0, 1)
        plt.legend(loc="upper center", fontsize=18)

        fig.add_subplot(122)
        sns.distplot(
            y_pred1_train,
            bins=100,
            color="blue",
            norm_hist=True,
            hist=False,
            kde_kws={"shade": True, "linewidth": width},
            label="Positive class",
        )
        sns.distplot(
            y_pred0_train,
            bins=100,
            color="green",
            norm_hist=True,
            hist=False,
            kde_kws={"shade": True, "linewidth": width},
            label="Negative class",
        )
        plt.xlim(0, 1)
        plt.legend(loc="upper center", fontsize=18)
        plt.show()

    def time_per_photon(self, X, iter: int = 10):
        model = self.model
        st = time.time()
        for _ in range(iter):
            y = model.predict(X)
        en = time.time()
        full_time = (en - st) / (iter * len(X))
        return full_time


def pairplot(training_data, training_labels):
    df = pd.DataFrame(training_data)
    df2 = pd.DataFrame(training_labels)
    dfn = pd.merge(df, df2, left_index=True, right_index=True)
    dfn = dfn.rename(
        {"0_x": "p_dphi", 1: "p_dtheta", 2: "x_dtheta", 3: "x_dphi", "0_y": "y"}, axis=1
    )
    dfn
    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(dfn, vars=dfn.columns[:-1], hue="y", diag_kind="kde")
    g.map_lower(sns.kdeplot)
    labels = g._legend_data.keys()
    handles = g._legend_data.values()
    g._legend.remove()
    g.fig.legend(labels=labels, handles=handles, fontsize=18, loc=7, frameon=False)


def accuracy_by_energy(model, data, name: str, range_):
    """Name is either 'e_energy' or 'b_energy' for electron or brem
    """

    df = data.sort_values(by=[name], ascending=True)
    split_df = np.array_split(df, range_)
    acc_complete = []
    for d in split_df:
        label_list = d["label"].to_numpy()
        new_df = d.drop(["label", "id", "e_energy", "b_energy"], axis=1)
        new_data = new_df.to_numpy()
        acc = model.eval(new_data, label_list)
        acc_complete.append(acc)

    xt = [
        "{} - {}".format(round(e_[name].min()), round(e_[name].max()))
        for e_ in split_df
    ]

    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    axs.bar(np.arange(range_), acc_complete, align="center", color="lightblue")
    axs.plot(np.arange(range_), acc_complete)
    axs.axhline(np.mean(acc_complete), color="blue")
    axs.set_ylim(
        np.min(acc_complete) - np.std(acc_complete),
        np.max(acc_complete) + np.std(acc_complete),
    )
    axs.set_xticks(np.arange(range_))
    axs.set_xticklabels(xt, rotation=75, fontsize=14)
    axs.set_ylabel("Accuracy", fontsize=18)
    axs.set_xlabel("Energies Range [Mev]", fontsize=18)
    plt.rcParams["ytick.labelsize"] = 16
    plt.show()
    return np.array(acc_complete), split_df


def time_by_energy(
    model, data, name: str, range_,
):
    """Name is either 'e_energy' or 'b_energy' for electron or brem
    """
    df = data.sort_values(by=[name], ascending=True)
    split_df = np.array_split(df, range_)
    time_list = []
    for i in range(10):
        timer = []
        for d in split_df:
            new_df = d.drop(["label", "id", "e_energy", "b_energy"], axis=1)
            new_data = new_df.to_numpy()
            len(new_data)
            # iterate 50 times before taking mean
            t = model.time_per_photon(new_data, 50)
            timer.append(t)
        time_list.append(timer)

    time_list = np.mean(time_list, axis=0)

    xt = [
        "{} - {}".format(round(e_[name].min()), round(e_[name].max()))
        for e_ in split_df
    ]

    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    axs.bar(np.arange(range_), time_list, align="center", color="lightblue")
    axs.plot(np.arange(range_), time_list)
    axs.axhline(np.mean(time_list), color="blue")
    axs.set_ylim(
        np.min(time_list) - np.std(time_list), np.max(time_list) + np.std(time_list)
    )
    axs.set_xticks(np.arange(range_))
    axs.set_xticklabels(xt, rotation=75, fontsize=14)
    axs.set_ylabel("Time per photon", fontsize=18)
    axs.set_xlabel("Energies Range [Mev]", fontsize=18)
    plt.rcParams["ytick.labelsize"] = 16
    plt.show()
    return time_list, split_df


if __name__ == "__main__":
    pass
    # data_interfaces = psiK.generate_data_interface("psiK_1000.root")
    # # data_interfaces = psiK.generate_data_interface("Bu2JpsiK_ee_mu1.1_1000_events.root")
    # data = psiK.generate_data_mixing(data_interfaces, sampling_frac=1)
    # (
    #     training_data,
    #     training_labels,
    # ) = psiK.generate_prepared_data(data, no_split = True)

    # np.random.seed(42)
    # tf.random.set_seed(42)
    # mlp = MLP_model(1000, 200)
    # history = mlp.train(
    #     training_data, training_labels, validation_data, validation_labels, 0
    # )
    # mlp.conf_mat_plot(
    #     training_data, training_labels, validation_data, validation_labels
    # )
    # mlp.plot_loss_acc(history)
    # mlp.plot_roc_curve(
    #     training_data, training_labels, validation_data, validation_labels
    # )
    # mlp.decision_function(
    #     training_data, training_labels, validation_data, validation_labels
    # )

    # svc = SVM_model(default=True)
    # svc.train(training_data, training_labels)
    # svc.conf_mat_plot(
    #     training_data, training_labels, validation_data, validation_labels
    # )
    # svc.plot_roc_curve(
    #     training_data, training_labels, validation_data, validation_labels
    # )
    # svc.decision_function(
    #     training_data, training_labels, validation_data, validation_labels
    # )

    # xgb = XGB_model()
    # xgb.train(training_data, training_labels)
    # xgb.conf_mat_plot(
    #     training_data, training_labels, validation_data, validation_labels
    # )
    # xgb.plot_roc_curve(
    #     training_data, training_labels, validation_data, validation_labels
    # )
    # xgb.decision_function(
    #     training_data, training_labels, validation_data, validation_labels
    # )
