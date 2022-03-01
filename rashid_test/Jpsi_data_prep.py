import uproot
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import awkward as ak
from copy import deepcopy
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

# from pydotplus import graph_from_dot_data
from IPython.display import Image

# from sklearn.tree import export_graphviz
from sklearn.ensemble import GradientBoostingClassifier

# import xgboost as xgb
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
pd.options.mode.chained_assignment = None
plt.rcParams["axes.xmargin"] = 0
# pd.set_option('display.max_rows', 400)


class DataPrep:
    def __init__(self, filename, features, N) -> None:
        self.filename = filename
        self.features = features
        self.N = N  # number of events to access in the file

    def extract_data(self):
        with uproot.open(self.filename) as file:
            tree = file["tuple/tuple"]
            Cluster_info = tree.arrays(self.features[6:], library="pd")
            ElectronTrack = tree.arrays(self.features[:6], library="pd")
            nElectronTracks = tree["nElectronTracks"].array()
            nCaloClusters = tree["nCaloClusters"].array()
            ElectronTrack_TYPE = tree["ElectronTrack_TYPE"].array()

            nET = []  # nET = no Electron Tracks
            for i in tqdm(range(self.N)):
                if nElectronTracks[i] < 1:  # Look for events with Electron Tracks
                    nET.append(i)
                elif (
                    ElectronTrack_TYPE[i, 0] != 3
                ):  # only accept Track TYPE of 3 (which is mostly in index 0 of ElectronTrack_Type array for a given event)
                    nET.append(i)
            nET = np.array(nET)

            even_events = np.arange(0, self.N, 2)
            odd_events = np.arange(1, self.N, 2)

            # Remove rows with no ElectronTracks
            mask1_indices_even = np.isin(even_events, nET)
            mask1_indices_odd = np.isin(odd_events, nET)
            even_events_after_mask1 = even_events[~mask1_indices_even]
            odd_events_after_mask1 = odd_events[~mask1_indices_odd]

            # Further Remove rows(events) with no CaloClusters
            nCI_even = (
                []
            )  # nCI = no Cluster Info, i.e. remove events with no cluster info
            for i in tqdm(even_events_after_mask1):
                if nCaloClusters[i] == 0:
                    nCI_even.append(i)
            nCI_even = np.array(nCI_even)

            nCI_odd = []
            for i in tqdm(odd_events_after_mask1):
                if nCaloClusters[i] == 0:
                    nCI_odd.append(i)
            nCI_odd = np.array(nCI_odd)

            mask2_indices_even = np.isin(even_events_after_mask1, nCI_even)
            mask2_indices_odd = np.isin(odd_events_after_mask1, nCI_odd)
            even_events_after_mask2 = even_events_after_mask1[~mask2_indices_even]
            odd_events_after_mask2 = odd_events_after_mask1[~mask2_indices_odd]

            # Remove the same odd and even entries (i.e 0 and 1 entry contain same info, thus if 0 is removed 1 must also be removed)
            match_mask = np.isin(even_events_after_mask2, odd_events_after_mask2 - 1)
            even_entries = even_events_after_mask2[match_mask]
            odd_entries = even_entries + 1

            # Extract the info from tree with correct indices
            ET_even = ElectronTrack.loc[even_entries, self.features[:6]]
            ET_odd = ElectronTrack.loc[odd_entries, self.features[:6]]
            CI_even = Cluster_info.loc[even_entries, self.features[6:]]
            CI_odd = Cluster_info.loc[odd_entries, self.features[6:]]

            # Remove nan's and inf's
            ET_even = ET_even[~ET_even.isin([np.nan, np.inf, -np.inf]).any(1)]
            ET_odd = ET_odd[~ET_odd.isin([np.nan, np.inf, -np.inf]).any(1)]
            CI_even = CI_even[~CI_even.isin([np.nan, np.inf, -np.inf]).any(1)]
            CI_odd = CI_odd[~CI_odd.isin([np.nan, np.inf, -np.inf]).any(1)]

            # Select only subentries with val 0 in pandas correspondong to TrackType 3 [because of first for loop]
            idx = pd.IndexSlice
            ET_even = ET_even.loc[idx[:, 0], :]
            ET_odd = ET_odd.loc[idx[:, 0], :]
            # print(ET_even.shape, ET_odd.shape)

            # Merging even and odd rows seperatly
            merge_even_BC_ET = pd.concat([ET_even, CI_even], axis=1).fillna(
                method="ffill"
            )
            merge_odd_BC_ET = pd.concat([ET_odd, CI_odd], axis=1).fillna(method="ffill")

            merge_even_BC_ET["label"] = 1
            merge_odd_BC_ET["label"] = 1
            df = pd.concat([merge_even_BC_ET, merge_odd_BC_ET], axis=0).sort_index()
        return (
            df.droplevel(level="subentry").reset_index().rename(columns={"entry": "id"})
        )

    def generate_data_mixing(self, df: pd.DataFrame, ratio_of_signal_to_background=1):
        ids = df["id"].unique().tolist()
        mixed_data = []
        columns_to_replace = [
            "ElectronTrack_PX",
            "ElectronTrack_PY",
            "ElectronTrack_PZ",
            "ElectronTrack_X",
            "ElectronTrack_Y",
            "ElectronTrack_Z",
        ]
        for id in tqdm(ids):
            running_df = df[df["id"] == id]
            running_df["label"] = 1
            sampled_df = df[df["id"] != id].sample(
                round(len(running_df.index.tolist()) * ratio_of_signal_to_background)
            )
            for column in columns_to_replace:
                sampled_df[column] = running_df[column].head(1).to_list()[0]
            sampled_df["label"] = 0
            combined_df = pd.concat([running_df, sampled_df])
            mixed_data.append(combined_df)

        return pd.concat(mixed_data)

    def train_validate_test_split(
        self, df: pd.DataFrame, test_ratio=0.1, validation_ratio=0.2, seed=None
    ):
        y_data = df["label"].to_numpy()
        X_data = df.drop(["label", "id"], axis=1).to_numpy()

        train_ratio = 1 - (test_ratio + validation_ratio)
        x_train, x_temp, y_train, y_temp = train_test_split(
            X_data,
            y_data,
            stratify=y_data,
            test_size=1 - train_ratio,
            random_state=seed,
        )

        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=test_ratio / (test_ratio + validation_ratio),
            shuffle=False,
        )

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    features = [
        "ElectronTrack_PX",
        "ElectronTrack_PY",
        "ElectronTrack_PZ",
        "ElectronTrack_X",
        "ElectronTrack_Y",
        "ElectronTrack_Z",
        "CaloCluster_E",
        "CaloCluster_X",
        "CaloCluster_Y",
        "CaloCluster_Z",
        "CaloCluster_Spread00",
        "CaloCluster_Spread01",
        "CaloCluster_Spread10",
        "CaloCluster_Spread11",
        "CaloCluster_Covariance00",
        "CaloCluster_Covariance01",
        "CaloCluster_Covariance02",
        "CaloCluster_Covariance10",
        "CaloCluster_Covariance11",
        "CaloCluster_Covariance12",
        "CaloCluster_Covariance20",
        "CaloCluster_Covariance21",
        "CaloCluster_Covariance22",
    ]

    fname = "data\\Bu2JpsiK_ee_1000_events.root"  #'data\\Bu2JpsiK_ee_1000_events.root'

    DP = DataPrep(fname, features, 2000)
    df = DP.extract_data()
    mixed_data_groups = DP.generate_data_mixing(df)
    (
        training_data,
        training_labels,
        validation_data,
        validation_labels,
    ) = DP.prepare_data(mixed_data_groups)

    #%% Test model with Calo Cluster info trained on electron gun dataset
    with open("ML_models\\xgb_model2_calo_info.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    train_pred = xgb_model.predict(training_data)
    val_pred = xgb_model.predict(validation_data)
    # test_pred =  xgb_model.predict(training_data1)
    train_score = accuracy_score(training_labels, train_pred)
    val_score = accuracy_score(validation_labels, val_pred)
    test_score = "Not explored here!"  # accuracy_score(training_labels1,test_pred)
    print(
        "train_score: ",
        train_score,
        " | ",
        "Val_score: ",
        val_score,
        " | ",
        "test_score: ",
        test_score,
    )

    feat_imp = pd.Series(xgb_model.feature_importances_, features).sort_values(
        ascending=False
    )
    feat_imp.plot(kind="bar", title="Feature Importances")
    plt.ylabel("Feature Importance Score")

    TN, FP, FN, TP = confusion_matrix(training_labels, train_pred).ravel()

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

