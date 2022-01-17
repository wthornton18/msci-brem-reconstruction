from copy import deepcopy
from ctypes import Union
from dataclasses import dataclass
from pprint import pprint
from time import time
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import RocCurveDisplay
from tqdm import tqdm
import uproot
from uproot import TBranch, ReadOnlyFile, TTree
import matplotlib.pyplot as plt
import numpy as np
from pylorentz import Momentum4
from tqdm.contrib import tzip
from itertools import compress, islice, chain
from random import sample


def ichunked(seq, chunksize):
    """Yields items from an iterator in iterable chunks."""
    it = iter(seq)
    while True:
        try:
            yield chain([next(it)], islice(it, chunksize - 1))
        except StopIteration:
            return


def chunked(seq, chunksize=2):
    """Yields items from an iterator in list chunks."""
    for chunk in ichunked(seq, chunksize):
        yield list(chunk)


class Vector3:
    def __init__(self, *x):
        if len(x) == 1 and isinstance(x[0], Vector3):
            x = x[0]
            self._values = np.array(x.components)
        elif len(x) == 3:
            self._values = np.array(list(x))
        else:
            raise TypeError("3-vector expects 3 values")

    def __repr__(self):
        if self._values.ndim == 1:
            pattern = "%g"
        else:
            pattern = "%r"

        return "%s(%s)" % (self.__class__.__name__, ", ".join([pattern % _ for _ in self._values]))

    def __sub__(self, other):

        vector = self.__class__(self)
        vector -= other
        return vector

    def __isub__(self, other):
        self._values = self.components - Vector3(*other).components
        return self

    def __iter__(self):
        return iter(self._values)

    @property
    def components(self):
        return self._values

    @property
    def trans(self):
        return np.sqrt(self._values[0] ** 2 + self._values[1] ** 2)

    @property
    def phi(self):
        return np.arctan2(self._values[1], self._values[0])

    @property
    def theta(self):
        return np.arctan2(self.trans, self._values[2])


class Position3(Vector3):
    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]


@dataclass(eq=False)
class DataInterface:
    _id: int
    electron_plus_momentum: Momentum4
    electron_plus_position: Position3
    electron_minus_momentum: Momentum4
    electron_minus_position: Position3
    brem_plus_momenta: List[Momentum4]
    brem_plus_positions: List[Position3]
    brem_minus_momenta: List[Momentum4]
    brem_minus_positions: List[Position3]
    std_electron_minus_momentum: Momentum4
    std_electron_plus_momentum: Momentum4
    jpsi_momentum: Momentum4
    b_momentum: Momentum4
    k_momentum: Momentum4

    def __init__(self, data):
        self.process_data(data)

    def process_data(self, data: Dict):
        self._id = data["id"]

        electron_plus_eta = data["e_plus_eta"]
        electron_plus_phi = data["e_plus_phi"]
        electron_plus_pt = data["e_plus_pt"]
        electron_plus_m = data["e_plus_m"]

        self.electron_plus_momentum = Momentum4.m_eta_phi_pt(
            electron_plus_m, electron_plus_eta, electron_plus_phi, electron_plus_pt
        )

        electron_plus_x = data["e_plus_x"]
        electron_plus_y = data["e_plus_y"]
        electron_plus_z = data["e_plus_z"]

        self.electron_plus_position = Position3(electron_plus_x, electron_plus_y, electron_plus_z)

        electron_minus_eta = data["e_minus_eta"]
        electron_minus_phi = data["e_minus_phi"]
        electron_minus_pt = data["e_minus_pt"]
        electron_minus_m = data["e_minus_m"]

        self.electron_minus_momentum = Momentum4.m_eta_phi_pt(
            electron_minus_m, electron_minus_eta, electron_minus_phi, electron_minus_pt
        )
        electron_minus_x = data["e_minus_x"]
        electron_minus_y = data["e_minus_y"]
        electron_minus_z = data["e_minus_z"]

        self.electron_minus_position = Position3(electron_minus_x, electron_minus_y, electron_minus_z)

        jpsi_eta = data["jpsi_eta"]
        jpsi_phi = data["jpsi_phi"]
        jpsi_pt = data["jpsi_pt"]
        jpsi_m = data["jpsi_m"]

        self.jpsi_momentum = Momentum4.m_eta_phi_pt(jpsi_m, jpsi_eta, jpsi_phi, jpsi_pt)

        b_eta = data["b_eta"]
        b_phi = data["b_phi"]
        b_pt = data["b_pt"]
        b_m = data["b_m"]

        self.b_momentum = Momentum4.m_eta_phi_pt(b_m, b_eta, b_phi, b_pt)

        k_eta = data["k_eta"]
        k_phi = data["k_phi"]
        k_pt = data["k_pt"]
        k_m = data["k_m"]

        self.k_momentum = Momentum4.m_eta_phi_pt(k_m, k_eta, k_phi, k_pt)

        std_e_plus_eta = data["std_electron_plus_eta"]
        std_e_plus_phi = data["std_electron_plus_phi"]
        std_e_plus_pt = data["std_electron_plus_pt"]
        std_e_plus_e = data["std_electron_plus_e"]

        self.std_electron_plus_momentum = Momentum4.e_eta_phi_pt(
            std_e_plus_e, std_e_plus_eta, std_e_plus_phi, std_e_plus_pt
        )

        std_e_minus_eta = data["std_electron_minus_eta"]
        std_e_minus_phi = data["std_electron_minus_phi"]
        std_e_minus_pt = data["std_electron_minus_pt"]
        std_e_minus_e = data["std_electron_minus_e"]

        self.std_electron_minus_momentum = Momentum4.e_eta_phi_pt(
            std_e_minus_e, std_e_minus_eta, std_e_minus_phi, std_e_minus_pt
        )

        brem_plus_positions = []
        brem_plus_momenta = []
        for brem_data in data["brem_plus_data"]:
            brem_x = brem_data["x"]
            brem_y = brem_data["y"]
            brem_z = brem_data["z"]
            brem_phi = brem_data["phi"]
            brem_eta = brem_data["eta"]
            brem_pt = brem_data["pt"]
            brem_e = brem_data["e"]
            brem_plus_positions.append(Position3(brem_x, brem_y, brem_z))
            brem_plus_momenta.append(Momentum4.e_eta_phi_pt(brem_e, brem_eta, brem_phi, brem_pt))
        self.brem_plus_momenta = brem_plus_momenta
        self.brem_plus_positions = brem_plus_positions

        brem_minus_positions = []
        brem_minus_momenta = []
        for brem_data in data["brem_minus_data"]:
            brem_x = brem_data["x"]
            brem_y = brem_data["y"]
            brem_z = brem_data["z"]
            brem_phi = brem_data["phi"]
            brem_eta = brem_data["eta"]
            brem_pt = brem_data["pt"]
            brem_e = brem_data["e"]
            brem_minus_positions.append(Position3(brem_x, brem_y, brem_z))
            brem_minus_momenta.append(Momentum4.e_eta_phi_pt(brem_e, brem_eta, brem_phi, brem_pt))
        self.brem_minus_momenta = brem_minus_momenta
        self.brem_minus_positions = brem_minus_positions

    def generate_data_slice(
        self,
        e_momentum: Momentum4,
        e_pos: Position3,
        brem_momenta: List[Momentum4],
        brem_positions: List[Position3],
        label: Optional[int] = None,
    ):
        out = []
        for (brem_momentum, brem_pos) in zip(brem_momenta, brem_positions):
            temp = {}
            dp: Momentum4 = e_momentum - brem_momentum
            dr: Position3 = e_pos - brem_pos
            temp = {"p_dphi": dp.phi, "p_dtheta": dp.theta, "x_dtheta": dr.theta, "x_dphi": dr.phi}
            if label is not None:
                temp.update({"id": self._id, "label": label})

            out.append(temp)

        return pd.DataFrame.from_records(out)

    def generate_external_data_slice(
        self, brem_momenta: List[Momentum4], brem_positions: List[Position3]
    ) -> pd.DataFrame:
        return pd.concat(
            [
                self.generate_data_slice(
                    self.electron_minus_momentum,
                    self.electron_minus_position,
                    brem_momenta,
                    brem_positions,
                    label=0,
                ),
                self.generate_data_slice(
                    self.electron_plus_momentum,
                    self.electron_plus_position,
                    brem_momenta,
                    brem_positions,
                    label=0,
                ),
            ]
        )

    def e_plus_data_slice(self, label=None) -> pd.DataFrame:
        return self.generate_data_slice(
            self.electron_plus_momentum,
            self.electron_plus_position,
            self.brem_plus_momenta,
            self.brem_plus_positions,
            label,
        )

    def e_minus_data_slice(self, label=None) -> pd.DataFrame:
        return self.generate_data_slice(
            self.electron_minus_momentum,
            self.electron_minus_position,
            self.brem_minus_momenta,
            self.brem_minus_positions,
            label,
        )

    def full_data_slice(self, label=None) -> pd.DataFrame:
        return pd.concat([self.e_plus_data_slice(label), self.e_minus_data_slice(label)])

    def full_brem_data(self) -> Tuple[List[Momentum4], List[Position3], int]:
        brem_momentum = deepcopy(self.brem_minus_momenta)
        brem_momentum.extend(self.brem_plus_momenta)
        brem_pos = deepcopy(self.brem_minus_positions)
        brem_pos.extend(self.brem_plus_positions)
        return (brem_momentum, brem_pos, self._id)

    @property
    def jpsi_noreco(self) -> Momentum4:
        return self.electron_plus_momentum + self.electron_minus_momentum

    @property
    def jpsi_stdreco(self) -> Momentum4:
        return self.std_electron_plus_momentum + self.std_electron_minus_momentum

    @property
    def jpsi_truereco(self) -> Momentum4:
        eplus_true_reco = self.electron_plus_momentum
        for brem_momentum in self.brem_plus_momenta:
            eplus_true_reco += brem_momentum
        eminus_true_reco = self.electron_minus_momentum
        for brem_momentum in self.brem_minus_momenta:
            eminus_true_reco += brem_momentum
        return eplus_true_reco + eminus_true_reco

    @property
    def b_noreco(self) -> Momentum4:
        return self.jpsi_noreco + self.k_momentum

    @property
    def b_stdreco(self) -> Momentum4:
        return self.jpsi_stdreco + self.k_momentum

    @property
    def b_truereco_from_electron(self) -> Momentum4:
        return self.jpsi_truereco + self.k_momentum

    @property
    def b_truereco(self) -> Momentum4:
        return self.k_momentum + self.jpsi_momentum


def generate_mass_curve(data: List[DataInterface]):
    hbmass_noreco = [m4.b_noreco.m[0] for m4 in data]
    hbmass_stdreco = [m4.b_stdreco.m[0] for m4 in data]
    hbmass_truereco = [m4.b_truereco_from_electron.m[0] for m4 in data]

    jspimass_noreco = [m4.jpsi_noreco.m[0] for m4 in data]
    jspimass_stdreco = [m4.jpsi_stdreco.m[0] for m4 in data]
    jspimass_truereco = [m4.jpsi_truereco.m[0] for m4 in data]


def get_names(filename: str):
    with uproot.open(filename) as file:
        print(file.keys())
        tree: Dict[str, TBranch] = file["tuple/tuple;1"]
        print(tree.keys())


def generate_brem_data(x, y, z, eta, phi, pt, e):
    x = list(x)
    y = list(y)
    z = list(z)
    eta = list(eta)
    phi = list(phi)
    pt = list(pt)
    e = list(e)
    return [
        {
            "x": data[0],
            "y": data[1],
            "z": data[2],
            "eta": data[3],
            "phi": data[4],
            "pt": data[5],
            "e": data[6],
        }
        for data in zip(x, y, z, eta, phi, pt, e)
    ]


def generate_data_interface(filename: str) -> List[DataInterface]:
    with uproot.open(filename) as file:
        tree: Dict[str, TBranch] = file["tuple/tuple;1"]
        data_interfaces = []
        for i, (e_plus_data, e_minus_data) in enumerate(
            chunked(
                zip(
                    tree["ElectronTrack_ETA"].array(),
                    tree["ElectronTrack_PHI"].array(),
                    tree["ElectronTrack_PT"].array(),
                    tree["electron_M"].array(),
                    tree["ElectronTrack_X"].array(),
                    tree["ElectronTrack_Y"].array(),
                    tree["ElectronTrack_Z"].array(),
                    tree["JPsi_ETA"].array(),
                    tree["JPsi_PHI"].array(),
                    tree["JPsi_PT"].array(),
                    tree["JPsi_M"].array(),
                    tree["B_ETA"].array(),
                    tree["B_PHI"].array(),
                    tree["B_PT"].array(),
                    tree["B_M"].array(),
                    tree["K_ETA"].array(),
                    tree["K_PHI"].array(),
                    tree["K_PT"].array(),
                    tree["K_M"].array(),
                    tree["StdBremReco_Electron_ETA"].array(),
                    tree["StdBremReco_Electron_PHI"].array(),
                    tree["StdBremReco_Electron_PT"].array(),
                    tree["StdBremReco_Electron_E"].array(),
                    tree["BremCluster_X"].array(),
                    tree["BremCluster_Y"].array(),
                    tree["BremCluster_Z"].array(),
                    tree["BremPhoton_PHI"].array(),
                    tree["BremPhoton_ETA"].array(),
                    tree["BremPhoton_PT"].array(),
                    tree["BremPhoton_E"].array(),
                    tree["nElectronTracks"].array(),
                    tree["ElectronTrack_TYPE"].array(),
                )
            )
        ):
            # if data[0][-2] < 1 or data[0][-1][0] != 3 or data[1][-2] < 1 or data[0][-1][0] != 3:
            # not a well constructed track
            #    continue

            if (
                e_plus_data[-2] < 1
                or e_plus_data[-1][0] != 3
                or e_minus_data[-2] < 1
                or e_minus_data[-1][0] != 3
            ):
                continue
            # e_plus_data = data[0]
            # e_minus_data = data[1]
            data_dict = {
                "id": i,
                "e_plus_eta": e_plus_data[0][0],
                "e_plus_phi": e_plus_data[1][0],
                "e_plus_pt": e_plus_data[2][0],
                "e_plus_m": e_plus_data[3],
                "e_plus_x": e_plus_data[4][0],
                "e_plus_y": e_plus_data[5][0],
                "e_plus_z": e_plus_data[6][0],
                "e_minus_eta": e_minus_data[0][0],
                "e_minus_phi": e_minus_data[1][0],
                "e_minus_pt": e_minus_data[2][0],
                "e_minus_m": e_minus_data[3],
                "e_minus_x": e_minus_data[4][0],
                "e_minus_y": e_minus_data[5][0],
                "e_minus_z": e_minus_data[6][0],
                "jpsi_eta": e_plus_data[7],
                "jpsi_phi": e_plus_data[8],
                "jpsi_pt": e_plus_data[9],
                "jpsi_m": e_plus_data[10],
                "b_eta": e_plus_data[11],
                "b_phi": e_plus_data[12],
                "b_pt": e_plus_data[13],
                "b_m": e_plus_data[14],
                "k_eta": e_plus_data[15],
                "k_phi": e_plus_data[16],
                "k_pt": e_plus_data[17],
                "k_m": e_plus_data[18],
                "std_electron_plus_eta": e_plus_data[19],
                "std_electron_plus_phi": e_plus_data[20],
                "std_electron_plus_pt": e_plus_data[21],
                "std_electron_plus_e": e_plus_data[22],
                "std_electron_minus_eta": e_minus_data[19],
                "std_electron_minus_phi": e_minus_data[20],
                "std_electron_minus_pt": e_minus_data[21],
                "std_electron_minus_e": e_minus_data[22],
                "brem_plus_data": generate_brem_data(*e_plus_data[23:30]),
                "brem_minus_data": generate_brem_data(*e_minus_data[23:30]),
            }
            d = DataInterface(deepcopy(data_dict))
            data_interfaces.append(d)
    return data_interfaces


def generate_data_mixing(data: List[DataInterface], sampling_frac: int = 2) -> pd.DataFrame:
    base_df = pd.concat([d.full_data_slice(label=1) for d in data])
    full = []
    brem_pos: List[Position3] = []
    brem_momentum: List[Momentum4] = []
    brem_id: List[int] = []

    for d in data:
        momentum, pos, i = d.full_brem_data()
        brem_pos.extend(pos)
        brem_momentum.extend(momentum)
        brem_id.extend([i] * len(momentum))
    i = 0
    for _id, group in base_df.groupby("id"):
        length = sum(map(lambda x: 1, filter(lambda _internal_id: _internal_id == _id, brem_id)))
        false_mask = list(map(lambda _internal_id: _internal_id != _id, brem_id))
        sample_pos = sample(list(compress(brem_pos, false_mask)), k=int(sampling_frac * length / 2))
        sample_momentum = sample(list(compress(brem_momentum, false_mask)), k=int(sampling_frac * length / 2))
        d = data[i]
        mixed_data = d.generate_external_data_slice(sample_momentum, sample_pos)
        full.extend([deepcopy(group), mixed_data])
        i += 1
    return pd.concat(full)


def generate_prepared_data(data: pd.DataFrame, split_frac: int = 0.9):
    label_list = data["label"].to_numpy()
    new_df = data.drop(["label", "id"], axis=1)
    new_data = new_df.to_numpy()
    indices = np.random.permutation(new_data.shape[0])
    i = int(split_frac * new_data.shape[0])
    training_idx, validation_idx = indices[:i], indices[i:]
    training_data, validation_data = new_data[training_idx, :], new_data[validation_idx, :]
    training_labels, validation_labels = label_list[training_idx], label_list[validation_idx]
    return training_data, training_labels, validation_data, validation_labels


def train_classifier(
    training_data: np.ndarray,
    train_labels: np.ndarray,
    validation_data: np.ndarray,
    validation_labels: np.ndarray,
) -> GradientBoostingClassifier:
    class Var:
        learning_rate = [0.1]
        n_estimators = [100]

    res = Var()

    gbc = GradientBoostingClassifier(learning_rate=res.learning_rate[0], n_estimators=res.n_estimators[0])
    gbc.fit(training_data, train_labels)
    return gbc


def plot_roc(
    gbc: GradientBoostingClassifier, training_data, train_labels, validation_data, validation_labels
):
    # fpr, tpr, thresholds = roc_curve(train_labels, gbc.decision_function(training_data))
    # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    roc_display_train = RocCurveDisplay.from_estimator(gbc, training_data, train_labels)
    roc_display_val = RocCurveDisplay.from_estimator(gbc, validation_data, validation_labels)
    roc_display_train.plot(ax=ax1, label="Train ")
    roc_display_val.plot(ax=ax2, label="Val")
    plt.show()


def plot_histo(
    gbc: GradientBoostingClassifier, training_data, train_labels, validation_data, validation_labels
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


if __name__ == "__main__":
    get_names("psiK_1000.root")
    out = generate_data_interface("psiK_1000.root")
    data = generate_data_mixing(out)
    training_data, training_labels, validation_data, validation_labels = generate_prepared_data(data)
    gbc = train_classifier(training_data, training_labels, validation_data, validation_labels)
    plot_histo(gbc, training_data, training_labels, validation_data, validation_labels)
