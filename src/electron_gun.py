import itertools
import math
from collections import defaultdict, namedtuple
from copy import deepcopy
from pprint import pprint
from re import purge, sub
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import uproot
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import RocCurveDisplay, roc_curve
from tqdm import tqdm
from tqdm.contrib import tzip
from uproot import TBranch, TTree
from utils import named_zip, Momentum4, chunked


class InitialModel:
    def __init__(self, filename, max_data) -> None:
        self.filename = filename
        self.max_data = max_data

    def prepare_data(self, df: pd.DataFrame, split_frac=0.9):
        label_list = df["label"].to_numpy()
        new_df = df.drop(["label", "id"], axis=1)
        new_data = new_df.to_numpy()
        indices = numpy.random.permutation(new_data.shape[0])
        i = int(split_frac * new_data.shape[0])
        print(new_df["BremCluster_E"].head(1))
        training_idx, validation_idx = indices[:i], indices[i:]
        training_data, validation_data = new_data[training_idx, :], new_data[validation_idx, :]
        training_labels, validation_labels = label_list[training_idx], label_list[validation_idx]
        return training_data, training_labels, validation_data, validation_labels

    def plot_histo(
        self, gbc: GradientBoostingClassifier, training_data, train_labels, validation_data, validation_labels
    ):
        train_acc = 100 * (sum(gbc.predict(training_data) == train_labels) / training_data.shape[0])
        val_acc = 100 * (sum(gbc.predict(validation_data) == validation_labels) / validation_data.shape[0])
        classifier_training_s = gbc.decision_function(training_data[train_labels == 1]).ravel()
        classifier_training_b = gbc.decision_function(training_data[train_labels == 0]).ravel()
        classifier_testing_s = gbc.decision_function(validation_data[validation_labels == 1]).ravel()
        classifier_testing_b = gbc.decision_function(validation_data[validation_labels == 0]).ravel()
        print(train_acc)
        print(val_acc)
        c_min = -10
        c_max = 10

        histo_training_s = np.histogram(classifier_training_s, bins=40, range=(c_min, c_max), density=True)
        histo_training_b = np.histogram(classifier_training_b, bins=40, range=(c_min, c_max), density=True)
        histo_testing_s = np.histogram(classifier_testing_s, bins=40, range=(c_min, c_max), density=True)
        histo_testing_b = np.histogram(classifier_testing_b, bins=40, range=(c_min, c_max), density=True)

        all_histos: List[np.histogram] = [
            histo_training_s,
            histo_training_b,
            histo_testing_s,
            histo_testing_b,
        ]
        h_max = max([histo[0].max() for histo in all_histos]) * 1.2
        h_min = max([histo[0].min() for histo in all_histos])
        bin_edges = histo_training_s[1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        ax1 = plt.subplot(111)

        # Draw solid histograms for the training data
        ax1.bar(
            bin_centers - bin_widths / 2.0,
            histo_training_s[0],
            facecolor="blue",
            linewidth=0,
            width=bin_widths,
            label="S (Train)",
            alpha=0.5,
        )
        ax1.bar(
            bin_centers - bin_widths / 2.0,
            histo_training_b[0],
            facecolor="red",
            linewidth=0,
            width=bin_widths,
            label="B (Train)",
            alpha=0.5,
        )

        # # Draw error-bar histograms for the testing data
        ax1.plot(bin_centers - bin_widths / 2, histo_testing_s[0], "bx", label="S (Test)")
        ax1.plot(bin_centers - bin_widths / 2, histo_testing_b[0], "rx", label="B (Test)")

        # Make a colorful backdrop to show the clasification regions in red and blue

        # Adjust the axis boundaries (just cosmetic)
        ax1.axis([c_min, c_max, h_min, h_max])

        # Make labels and title
        plt.title("Classification with scikit-learn")
        plt.xlabel("Classifier, GBC")
        plt.ylabel("Counts/Bin")
        # Make legend with small font
        legend = ax1.legend(loc="upper center", shadow=True, ncol=2)
        for alabel in legend.get_texts():
            alabel.set_fontsize("small")

        plt.show()

    def plot_roc(
        self, gbc: GradientBoostingClassifier, training_data, train_labels, validation_data, validation_labels
    ):
        # fpr, tpr, thresholds = roc_curve(train_labels, gbc.decision_function(training_data))
        # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        roc_display_train = RocCurveDisplay.from_estimator(gbc, training_data, train_labels)
        roc_display_val = RocCurveDisplay.from_estimator(gbc, validation_data, validation_labels)
        roc_display_train.plot(ax=ax1, label="Train ")
        roc_display_val.plot(ax=ax2, label="Val")
        plt.show()

    def generate_bins(self, training_data, accuracy):
        electron_energies = []
        for tr in training_data:
            electron_energies.append(tr[6])
        min_ee = np.min(electron_energies)
        max_ee = np.max(electron_energies)
        bins = defaultdict(int)
        total = defaultdict(int)
        energ_ranges = (max_ee - min_ee) / 40
        for acc, energ in zip(accuracy, electron_energies):
            i = energ // energ_ranges
            if acc == 1:
                bins[i] += 1
            total[i] += 1
        out = []
        for i, acc in bins.items():
            tot = total.get(i)
            out.append(acc / tot)
        return out, energ_ranges

    def calculate_energy_recon(
        self, gbc: GradientBoostingClassifier, training_data, train_labels, validation_data, validation_labels
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        tr_accuracy = gbc.predict(training_data) == train_labels
        val_accuracy = gbc.predict(validation_data) == validation_labels
        tr_out, tr_energy = self.generate_bins(training_data, tr_accuracy)
        val_out, val_energy = self.generate_bins(validation_data, val_accuracy)
        ax1.plot(tr_energy * np.arange(1, len(tr_out) + 1), tr_out)
        ax2.plot(val_energy * np.arange(1, len(val_out) + 1), val_out)
        plt.show()

    def train_model(
        self, training_data, train_labels, validation_data, validation_labels
    ) -> GradientBoostingClassifier:

        total_size = training_data.shape[0] + validation_data.shape[0]
        # res = skopt.gp_minimize(fn_optimise, dimensions=search_space)
        class Var:
            learning_rate = [0.1]
            n_estimators = [100]

        res = Var()
        gbc = GradientBoostingClassifier(learning_rate=res.learning_rate[0], n_estimators=res.n_estimators[0])
        gbc.fit(training_data, train_labels)
        return gbc
        """
        #model.fit(
            x=training_data,
            y=train_labels,
            epochs=10,
            validation_data=(validation_data, validation_labels)
        #)
        """

    def get_names(self):
        with uproot.open(self.filename) as file:
            tree: Dict[str, TBranch] = file["tuple/tuple;3"]
            print(tree.keys())

    def generate_data_mapping(self):

        with uproot.open(self.filename) as file:
            tree: Dict[str, TBranch] = file["tuple/tuple;4"]
            electron_data_sets = []
            i = 0
            for data in tzip(
                tree["ElectronTrack_PX"].array(),
                tree["ElectronTrack_PY"].array(),
                tree["ElectronTrack_PZ"].array(),
                tree["ElectronTrack_X"].array(),
                tree["ElectronTrack_Y"].array(),
                tree["ElectronTrack_Z"].array(),
                tree["BremCluster_E"].array(),
                tree["BremCluster_X"].array(),
                tree["BremCluster_Y"].array(),
                tree["BremCluster_Z"].array(),
            ):
                if i > self.max_data:
                    break
                else:
                    i += 1
                boolean_filter = list(
                    map(
                        lambda x: x[0] and x[1] and x[2],
                        zip(
                            map(math.isfinite, data[0]),
                            map(math.isfinite, data[1]),
                            map(math.isfinite, data[2]),
                        ),
                    )
                )
                first_filtering = {
                    "ElectronTrack_PX": list(filter(math.isfinite, data[0])),
                    "ElectronTrack_PY": list(filter(math.isfinite, data[1])),
                    "ElectronTrack_PZ": list(filter(math.isfinite, data[2])),
                }
                skip_data = False
                second_filtering = {}
                for k, v in first_filtering.items():
                    if len(v) == 0:
                        skip_data = True
                        break
                    else:
                        second_filtering[k] = v[0]
                if skip_data:
                    continue
                second_filtering.update(
                    {
                        "ElectronTrack_X": list(itertools.compress(data[3], boolean_filter))[0],
                        "ElectronTrack_Y": list(itertools.compress(data[4], boolean_filter))[0],
                        "ElectronTrack_Z": list(itertools.compress(data[5], boolean_filter))[0],
                        "BremCluster_E": list(data[6]),
                        "BremCluster_X": list(data[7]),
                        "BremCluster_Y": list(data[8]),
                        "BremCluster_Z": list(data[9]),
                    }
                )
                electron_data_sets.append(deepcopy(second_filtering))
        switched_electron_data = []
        for i, data in enumerate(electron_data_sets):
            temp = {"id": i}
            temp.update({k: v if "ElectronTrack" in k else None for k, v in data.items()})
            brem_e, brem_x, brem_y, brem_z = (
                data["BremCluster_E"],
                data["BremCluster_X"],
                data["BremCluster_Y"],
                data["BremCluster_Z"],
            )
            for brem_data in zip(brem_e, brem_x, brem_y, brem_z):
                refactored_data = deepcopy(temp)
                refactored_data.update(
                    {
                        "BremCluster_E": brem_data[0],
                        "BremCluster_X": brem_data[1],
                        "BremCluster_Y": brem_data[2],
                        "BremCluster_Z": brem_data[3],
                    }
                )
                switched_electron_data.append(refactored_data)

        df = pd.DataFrame.from_dict(switched_electron_data)
        return df

    def generate_data_mixing(self, df: pd.DataFrame):
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
            sampled_df = df[df["id"] != id].sample(len(running_df.index.tolist()))
            for column in columns_to_replace:
                sampled_df[column] = running_df[column].head(1).to_list()[0]
            sampled_df["label"] = 0
            combined_df = pd.concat([running_df, sampled_df])
            mixed_data.append(combined_df)
        data = pd.concat(mixed_data)
        return data


def generate_electron_curve(filename: str) -> List[float]:
    with uproot.open(filename) as file:
        tree: Dict[str, TBranch] = file["tuple/tuple;1"]
        out = []
        for e_cluster, e_photon, n_brem in zip(
            tree["CaloCluster_E"].array(), tree["BremPhoton_E"].array(), tree["nBremPhotons"].array()
        ):
            e_cluster_sum = np.sum(e_cluster)
            e_photon_sum = np.sum(e_photon)
            if e_photon_sum != 0:
                out.append(e_cluster_sum / e_photon_sum)
        return out


def generate_electron_momentum_curve(
    filename: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    ovx, ovy, ovz = generate_ov_distribution(filename)
    ovx = np.mean(ovx)
    ovy = np.mean(ovy)
    ovz = np.mean(ovz)
    with uproot.open(filename) as file:
        tree: Dict[str, TBranch] = file["tuple/tuple;1"]
        total_e = []
        total_px = []
        total_py = []
        total_pz = []
        data_tuple = namedtuple(
            "data_tuple", ["c_e", "x", "y", "z", "px", "py", "pz", "p_e", "ntracks", "track_type"]
        )
        for data in named_zip(
            data_tuple,
            tree["BremCluster_E"].array(),
            tree["BremCluster_X"].array(),
            tree["BremCluster_Y"].array(),
            tree["BremCluster_Z"].array(),
            tree["BremPhoton_PX"].array(),
            tree["BremPhoton_PY"].array(),
            tree["BremPhoton_PZ"].array(),
            tree["BremPhoton_E"].array(),
            tree["nElectronTracks"].array(),
            tree["ElectronTrack_TYPE"].array(),
        ):
            if data.ntracks < 1 or data.track_type[0] != 3:
                continue
            true_brem_momenta = Momentum4(0, 0, 0, 0)
            reco_brem_momenta = Momentum4(0, 0, 0, 0)
            for (e, px, py, pz) in zip(data.p_e, data.px, data.py, data.pz):
                brem_momenta = Momentum4(e, px, py, pz)
                true_brem_momenta += brem_momenta
            if len(data.c_e) > 1:
                continue
            for (e, x, y, z) in zip(data.c_e, data.x, data.y, data.z):
                x -= ovx
                y -= ovy
                z -= ovz
                mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                if mag == 0:
                    continue
                ratio = (1 / mag) * e
                brem_momenta = Momentum4(e, ratio * x, ratio * y, ratio * z)
                reco_brem_momenta += brem_momenta
            if true_brem_momenta.e != 0:
                total_e.append(reco_brem_momenta.e / true_brem_momenta.e)
            if true_brem_momenta.p_x != 0:
                total_px.append(reco_brem_momenta.p_x / true_brem_momenta.p_x)
            if true_brem_momenta.p_y != 0:
                total_py.append(reco_brem_momenta.p_y / true_brem_momenta.p_y)
            if true_brem_momenta.p_z != 0:
                total_pz.append(reco_brem_momenta.p_z / true_brem_momenta.p_z)
        return total_e, total_px, total_py, total_pz


def generate_electron_momentum_delta(
    filename: str,
) -> List[Momentum4]:
    ovx, ovy, ovz = generate_ov_distribution(filename)
    ovx = np.mean(ovx)
    ovy = np.mean(ovy)
    ovz = np.mean(ovz)
    with uproot.open(filename) as file:
        tree: Dict[str, TBranch] = file["tuple/tuple;1"]
        delta_momenta = []
        data_tuple = namedtuple("data_tuple", ["c_e", "x", "y", "z", "px", "py", "pz", "p_e"])
        for data in named_zip(
            data_tuple,
            tree["CaloCluster_E"].array(),
            tree["CaloCluster_X"].array(),
            tree["CaloCluster_Y"].array(),
            tree["CaloCluster_Z"].array(),
            tree["BremPhoton_PX"].array(),
            tree["BremPhoton_PY"].array(),
            tree["BremPhoton_PZ"].array(),
            tree["BremPhoton_E"].array(),
        ):
            true_brem_momenta = Momentum4(0, 0, 0, 0)
            reco_brem_momenta = Momentum4(0, 0, 0, 0)
            for (e, px, py, pz) in zip(data.p_e, data.px, data.py, data.pz):
                brem_momenta = Momentum4(e, px, py, pz)
                true_brem_momenta += brem_momenta
            for (e, x, y, z) in zip(data.c_e, data.x, data.y, data.z):
                x -= ovx
                y -= ovy
                z -= ovz
                mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                if mag == 0:
                    continue
                ratio = (1 / mag) * e
                brem_momenta = Momentum4(e, ratio * x, ratio * y, ratio * z)
                reco_brem_momenta += brem_momenta
            if (
                true_brem_momenta.e != 0
                or true_brem_momenta.p_x != 0
                or true_brem_momenta.p_y != 0
                or true_brem_momenta.p_z != 0
            ):
                delta_momenta.append(true_brem_momenta - reco_brem_momenta)
        return delta_momenta


def generate_ov_distribution(filename: str):
    with uproot.open(filename) as file:
        tree: Dict[str, TBranch] = file["tuple/tuple;1"]
        OVX = []
        OVY = []
        OVZ = []
        data_tuple = namedtuple("data_tuple", ["ovx", "ovy", "ovz"])
        for data in named_zip(
            data_tuple,
            tree["BremPhoton_OVX"].array(),
            tree["BremPhoton_OVY"].array(),
            tree["BremPhoton_OVZ"].array(),
        ):
            for (ovx, ovy, ovz) in zip(data.ovx, data.ovy, data.ovz):
                if ovz > 5000:
                    continue
                OVX.append(ovx)
                OVY.append(ovy)
                OVZ.append(ovz)
    return OVX, OVY, OVZ


if __name__ == "__main__":
    print(generate_electron_curve("./1000ev.root"))
