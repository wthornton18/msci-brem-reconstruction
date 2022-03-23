import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import RocCurveDisplay, confusion_matrix
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import seaborn as sns
import psiK


class MLP_model:
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, x_train, y_train, x_test, y_test, verbose=2):
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
        y_pred1_val = self.model.predict_proba(validation_data[ind1_val])[:, 1]
        y_pred0_val = self.model.predict_proba(validation_data[ind0_val])[:, 1]

        ind1_train = list(np.where(training_labels == 1))[0]
        ind0_train = list(np.where(training_labels == 0))[0]
        y_pred1_train = self.model.predict_proba(training_data[ind1_train])[:, 1]
        y_pred0_train = self.model.predict_proba(training_data[ind0_train])[:, 1]

        width = 3
        fig = plt.figure(figsize=(15, 10))
        fig.add_subplot(121)
        sns.histplot(
            y_pred1_train,
            bins=100,
            color="blue",
            kde=True,
            shade=True,
            lindewidth=width,
            label="Positive class",
        )
        sns.histplot(
            y_pred0_train,
            bins=100,
            color="blue",
            kde=True,
            shade=True,
            lindewidth=width,
            label="Negative class",
        )
        plt.xlim(0, 1)
        plt.title("Tranining data decision plot")
        plt.legend(loc="upper center", fontsize=18)

        fig.add_subplot(122)
        sns.histplot(
            y_pred1_val,
            bins=100,
            color="blue",
            kde=True,
            shade=True,
            lindewidth=width,
            label="Positive class",
        )
        sns.histplot(
            y_pred0_val,
            bins=100,
            color="blue",
            kde=True,
            shade=True,
            lindewidth=width,
            label="Negative class",
        )
        plt.xlim(0, 1)
        plt.title("Validation data decision plot")
        plt.legend(loc="upper center", fontsize=18)
        plt.show()


if __name__ == "__main__":
    data_interfaces = psiK.generate_data_interface("psiK_1000.root")
    # data_interfaces = psiK.generate_data_interface("Bu2JpsiK_ee_mu1.1_1000_events.root")
    data = psiK.generate_data_mixing(data_interfaces, sampling_frac=1)
    (
        training_data,
        training_labels,
        validation_data,
        validation_labels,
    ) = psiK.generate_prepared_data(data)

    classifier = psiK.train_xgboost(
        training_data, training_labels, validation_data, validation_labels
    )
    fig, axs = plt.subplots(2, 1, figsize=(18, 18))
    plot_confusion_matrix(
        classifier, training_data, training_labels, ax=axs[0], cmap=plt.cm.Blues
    )
    plot_confusion_matrix(
        classifier, validation_data, validation_labels, ax=axs[1], cmap=plt.cm.Blues
    )
    axs[0].set_title("Training Data on XGBoost")
    axs[1].set_title("Validation Data on XGBoost")
    plt.show()
    # plot_masses(classifier, data_interfaces, 2000)

    np.random.seed(42)
    tf.random.set_seed(42)
    mlp = MLP_model(1000, 200)
    history = mlp.train(
        training_data, training_labels, validation_data, validation_labels, 0
    )
    mlp.conf_mat_plot(
        training_data, training_labels, validation_data, validation_labels
    )
    mlp.plot_loss_acc(history)
    mlp.plot_roc_curve(
        training_data, training_labels, validation_data, validation_labels
    )
    mlp.decision_function(
        training_data, training_labels, validation_data, validation_labels
    )

