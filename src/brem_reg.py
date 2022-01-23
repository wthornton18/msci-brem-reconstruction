from dataclasses import dataclass

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRFRegressor
from utils import Position3, Momentum4, named_zip
from typing import List, Dict, Tuple
from collections import namedtuple
import numpy as np
import uproot
from uproot import TBranch
from sklearn.metrics import explained_variance_score
from torch import nn


device = "cpu"


@dataclass
class SubDataInterface:
    brem_pos: Position3
    brem_e: float
    truebrem_momentum: Momentum4
    electron_momentum: Momentum4
    brem_ov: Position3

    @property
    def x(self):
        a = [
            self.brem_pos.x,
            self.brem_pos.y,
            self.brem_pos.z,
            self.brem_e,
            self.electron_momentum.p_x,
            self.electron_momentum.p_y,
            self.electron_momentum.p_z,
            self.electron_momentum.e,
        ]
        return a

    @property
    def y_momentum(self):
        return [
            self.truebrem_momentum.p_x,
            self.truebrem_momentum.p_y,
            self.truebrem_momentum.p_z,
            self.truebrem_momentum.e,
        ]

    @property
    def y_ov(self):
        return [self.brem_ov.x, self.brem_ov.y, self.brem_ov.z]


def generate_regressor_datainterfaces(filename: str) -> List[SubDataInterface]:

    data_tuple = namedtuple(
        "data_tuple",
        [
            "brem_x",
            "brem_y",
            "brem_z",
            "brem_e",
            "truebrem_px",
            "truebrem_py",
            "truebrem_pz",
            "truebrem_e",
            "truebrem_ovx",
            "truebrem_ovy",
            "truebrem_ovz",
            "e_px",
            "e_py",
            "e_pz",
            "e_m",
            "n_brem",
            "n_tracks",
            "type",
        ],
    )

    data_interfaces = []
    with uproot.open(filename) as file:
        tree: Dict[str, TBranch] = file["tuple/tuple;1"]
        for e_data in named_zip(
            data_tuple,
            tree["BremCluster_X"].array(),
            tree["BremCluster_Y"].array(),
            tree["BremCluster_Z"].array(),
            tree["BremCluster_E"].array(),
            tree["BremPhoton_PX"].array(),
            tree["BremPhoton_PY"].array(),
            tree["BremPhoton_PZ"].array(),
            tree["BremPhoton_E"].array(),
            tree["BremPhoton_OVX"].array(),
            tree["BremPhoton_OVY"].array(),
            tree["BremPhoton_OVZ"].array(),
            tree["ElectronTrack_PX"].array(),
            tree["ElectronTrack_PY"].array(),
            tree["ElectronTrack_PZ"].array(),
            tree["electron_M"].array(),
            tree["nBremPhotons"].array(),
            tree["nElectronTracks"].array(),
            tree["ElectronTrack_TYPE"].array(),
        ):
            if e_data.n_tracks < 1 or e_data.n_brem < 1 or e_data.type[0] != 3:
                continue

            e_momentum = Momentum4.m_px_py_pz(e_data.e_m, e_data.e_px[0], e_data.e_py[0], e_data.e_pz[0])
            for (x, y, z, e, px, py, pz, true_e, ovx, ovy, ovz) in zip(
                list(e_data.brem_x),
                list(e_data.brem_y),
                list(e_data.brem_z),
                list(e_data.brem_e),
                list(e_data.truebrem_px),
                list(e_data.truebrem_py),
                list(e_data.truebrem_pz),
                list(e_data.truebrem_e),
                list(e_data.truebrem_ovx),
                list(e_data.truebrem_ovy),
                list(e_data.truebrem_ovz),
            ):
                if ovz > 5000:
                    continue
                pos = Position3(x, y, z)
                ov = Position3(ovx, ovy, ovz)
                momentum = Momentum4(true_e, px, py, pz)
                data_interfaces.append(
                    SubDataInterface(
                        brem_pos=pos,
                        brem_e=e,
                        truebrem_momentum=momentum,
                        electron_momentum=e_momentum,
                        brem_ov=ov,
                    )
                )
    return data_interfaces


def generate_data_mixing(out: List[SubDataInterface]):
    print(len(out))

    x = np.array([d.x for d in out])
    y = np.array([d.y_ov for d in out])
    print(x.shape)
    print(y.shape)
    return x, y


def generate_prepared_data(x: np.ndarray, y: np.ndarray, split_frac: int = 0.9):
    indicies = np.random.permutation(x.shape[0])
    i = int(split_frac * x.shape[0])
    training_idx, validation_idx = indicies[:i], indicies[i:]
    training_data, validation_data = x[training_idx, :], x[validation_idx, :]
    training_labels, validation_labels = y[training_idx, :], y[validation_idx, :]
    print(training_data.shape)
    print(validation_data.shape)
    print(training_labels.shape)
    print(validation_labels.shape)
    return training_data, training_labels, validation_data, validation_labels


def train_ndregressor(training_data, training_labels, validation_data, validation_labels):
    mxgbr = MultiOutputRegressor(XGBRFRegressor()).fit(training_data, training_labels)
    print(np.mean((mxgbr.predict(training_data) - training_labels) ** 2, axis=0))
    print(
        explained_variance_score(mxgbr.predict(training_data), training_labels, multioutput="uniform_average")
    )
    print(
        explained_variance_score(
            mxgbr.predict(validation_data), validation_labels, multioutput="uniform_average"
        )
    )


if __name__ == "__main__":
    out = generate_regressor_datainterfaces("psiK_1000.root")
    x, y = generate_data_mixing(out)
    training_data, training_labels, validation_data, validation_labels = generate_prepared_data(x, y)
    train_ndregressor(training_data, training_labels, validation_data, validation_labels)
