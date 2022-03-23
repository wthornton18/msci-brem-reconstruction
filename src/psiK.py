from copy import copy, deepcopy
from ctypes import Union
from dataclasses import dataclass
from functools import partial
import math
from pprint import pprint
from time import time
import scipy as sp
from scipy.stats import mode
from typing import Dict, Iterable, List, Optional, Tuple
import matplotlib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import RocCurveDisplay, confusion_matrix
from tqdm import tqdm
import uproot
from uproot import TBranch, ReadOnlyFile, TTree
import matplotlib.pyplot as plt
import numpy as np
from tqdm.contrib import tzip
from itertools import compress, islice, chain
from random import sample
from collections import namedtuple
from utils import Position3, chunked, Momentum4, named_zip, data
from scipy.optimize import minimize, basinhopping
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold


@dataclass(eq=False)
class DataInterface:
    _id: int
    electron_plus_momentum: Momentum4
    electron_plus_position: Position3
    electron_minus_momentum: Momentum4
    electron_minus_position: Position3
    brem_plus_momenta: List[Momentum4]
    true_brem_plus_momenta: List[Momentum4]
    brem_plus_positions: List[Position3]
    brem_minus_momenta: List[Momentum4]
    true_brem_minus_momenta: List[Momentum4]
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

        electron_plus_px = data["e_plus_px"]
        electron_plus_py = data["e_plus_py"]
        electron_plus_pz = data["e_plus_pz"]
        electron_plus_m = data["e_plus_m"]

        self.electron_plus_momentum = Momentum4.m_px_py_pz(
            electron_plus_m, electron_plus_px, electron_plus_py, electron_plus_pz
        )

        electron_plus_x = data["e_plus_x"]
        electron_plus_y = data["e_plus_y"]
        electron_plus_z = data["e_plus_z"]

        self.electron_plus_position = Position3(
            electron_plus_x, electron_plus_y, electron_plus_z
        )

        electron_minus_px = data["e_minus_px"]
        electron_minus_py = data["e_minus_py"]
        electron_minus_pz = data["e_minus_pz"]
        electron_minus_m = data["e_minus_m"]

        self.electron_minus_momentum = Momentum4.m_px_py_pz(
            electron_minus_m, electron_minus_px, electron_minus_py, electron_minus_pz
        )
        electron_minus_x = data["e_minus_x"]
        electron_minus_y = data["e_minus_y"]
        electron_minus_z = data["e_minus_z"]

        self.electron_minus_position = Position3(
            electron_minus_x, electron_minus_y, electron_minus_z
        )

        jpsi_px = data["jpsi_px"]
        jpsi_py = data["jpsi_py"]
        jpsi_pz = data["jpsi_pz"]
        jpsi_m = data["jpsi_m"]

        self.jpsi_momentum = Momentum4.m_px_py_pz(jpsi_m, jpsi_px, jpsi_py, jpsi_pz)

        b_px = data["b_px"]
        b_py = data["b_py"]
        b_pz = data["b_pz"]
        b_m = data["b_m"]

        self.b_momentum = Momentum4.m_px_py_pz(b_m, b_px, b_py, b_pz)

        k_px = data["k_px"]
        k_py = data["k_py"]
        k_pz = data["k_pz"]
        k_m = data["k_m"]

        self.k_momentum = Momentum4.m_px_py_pz(k_m, k_px, k_py, k_pz)

        std_e_plus_px = data["std_electron_plus_px"]
        std_e_plus_py = data["std_electron_plus_py"]
        std_e_plus_pz = data["std_electron_plus_pz"]
        std_e_plus_e = data["std_electron_plus_e"]

        self.std_electron_plus_momentum = Momentum4(
            std_e_plus_e, std_e_plus_px, std_e_plus_py, std_e_plus_pz
        )

        std_e_minus_px = data["std_electron_minus_px"]
        std_e_minus_py = data["std_electron_minus_py"]
        std_e_minus_pz = data["std_electron_minus_pz"]
        std_e_minus_e = data["std_electron_minus_e"]

        self.std_electron_minus_momentum = Momentum4(
            std_e_minus_e, std_e_minus_px, std_e_minus_py, std_e_minus_pz
        )

        brem_plus_positions = []
        brem_plus_momenta = []
        true_brem_plus_momenta = []
        for brem_data in data["brem_plus_data"]:
            brem_x = brem_data["x"]
            brem_y = brem_data["y"]
            brem_z = brem_data["z"]
            brem_py = brem_data["py"]
            brem_px = brem_data["px"]
            brem_pz = brem_data["pz"]
            brem_e = brem_data["e"]
            brem_plus_momenta.append(
                reconstruct_brem(brem_x, brem_y, brem_z, brem_data["cluster_e"])
            )
            brem_plus_positions.append(Position3(brem_x, brem_y, brem_z))
            true_brem_plus_momenta.append(Momentum4(brem_e, brem_px, brem_py, brem_pz))
        self.brem_plus_momenta = brem_plus_momenta
        self.brem_plus_positions = brem_plus_positions
        self.true_brem_plus_momenta = true_brem_plus_momenta
        brem_minus_positions = []
        brem_minus_momenta = []
        true_brem_minus_momenta = []
        for brem_data in data["brem_minus_data"]:
            brem_x = brem_data["x"]
            brem_y = brem_data["y"]
            brem_z = brem_data["z"]
            brem_py = brem_data["py"]
            brem_px = brem_data["px"]
            brem_pz = brem_data["pz"]
            brem_e = brem_data["e"]
            brem_minus_momenta.append(
                reconstruct_brem(brem_x, brem_y, brem_z, brem_data["cluster_e"])
            )
            brem_minus_positions.append(Position3(brem_x, brem_y, brem_z))
            true_brem_minus_momenta.append(Momentum4(brem_e, brem_px, brem_py, brem_pz))
        self.brem_minus_momenta = brem_minus_momenta
        self.brem_minus_positions = brem_minus_positions
        self.true_brem_minus_momenta = true_brem_minus_momenta

    def generate_data_slice(
        self,
        e_momentum: Momentum4,
        e_pos: Position3,
        brem_momenta: List[Momentum4],
        brem_positions: List[Position3],
        label: Optional[int] = None,
        cutoff: float = None,
    ):
        if cutoff is None:
            cutoff = 0
        out = []
        for (brem_momentum, brem_pos) in zip(brem_momenta, brem_positions):
            if brem_momentum.e > cutoff:
                temp = {}
                dp: Momentum4 = e_momentum - brem_momentum
                dr: Position3 = e_pos - brem_pos
                temp = {
                    "p_dphi": dp.phi,
                    "p_dtheta": dp.theta,
                    "x_dtheta": dr.theta,
                    "x_dphi": dr.phi,
                    "e_energy": e_momentum.e,  ### Modified
                    "b_energy": brem_momentum.e,  ### Modified
                }
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
        return pd.concat(
            [self.e_plus_data_slice(label), self.e_minus_data_slice(label)]
        )

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
        eplus_true_reco = copy(self.electron_plus_momentum)
        for brem_momentum in self.true_brem_plus_momenta:
            eplus_true_reco += brem_momentum
        eminus_true_reco = copy(self.electron_minus_momentum)
        for brem_momentum in self.true_brem_minus_momenta:
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

    def jpsi_ourreco(
        self, classifier: XGBClassifier, cutoff: Optional[float] = None
    ) -> Momentum4:
        full_brem_pos = deepcopy(self.brem_minus_positions)
        full_brem_momentum = deepcopy(self.brem_minus_momenta)
        full_brem_pos.extend(self.brem_plus_positions)
        full_brem_momentum.extend(self.brem_plus_momenta)
        plus_df = self.generate_data_slice(
            self.electron_plus_momentum,
            self.electron_plus_position,
            full_brem_momentum,
            full_brem_pos,
            cutoff=cutoff,
        )

        minus_df = self.generate_data_slice(
            self.electron_minus_momentum,
            self.electron_minus_position,
            full_brem_momentum,
            full_brem_pos,
            cutoff=cutoff,
        )
        e_plus_reco = copy(self.electron_plus_momentum)
        e_minus_reco = self.electron_minus_momentum
        plus_arr = plus_df.to_numpy()
        minus_arr = minus_df.to_numpy()
        if len(plus_arr) > 0:
            plus_predictions: List[bool] = classifier.predict(plus_arr) == 1
            plus_brem_momentum: List[Momentum4] = list(
                compress(full_brem_momentum, plus_predictions)
            )
            for brem_momentum in plus_brem_momentum:
                e_plus_reco += brem_momentum

        if len(minus_arr) > 0:
            minus_predictions: List[bool] = classifier.predict(minus_arr) == 1
            minus_brem_momentum: List[Momentum4] = list(
                compress(full_brem_momentum, minus_predictions)
            )
            for brem_momentum in copy(minus_brem_momentum):
                e_minus_reco += brem_momentum
        return e_plus_reco + e_minus_reco

    def b_ourreco(
        self, classifier: XGBClassifier, cutoff: Optional[float] = None
    ) -> Momentum4:
        return self.jpsi_ourreco(classifier=classifier, cutoff=cutoff) + self.k_momentum


def get_names(filename: str):
    from pprint import pprint

    with uproot.open(filename) as file:
        print(file.keys())
        tree: Dict[str, TBranch] = file["tuple/tuple;1"]
        pprint(tree.keys())


def generate_brem_data(x, y, z, cluster_e, px, py, pz, e, ovz):
    x = list(x)
    y = list(y)
    z = list(z)
    cluster_e = list(cluster_e)
    px = list(px)
    py = list(py)
    pz = list(pz)
    e = list(e)
    ovz = list(ovz)
    return [
        {
            "x": data[0],
            "y": data[1],
            "z": data[2],
            "cluster_e": data[3],
            "px": data[4],
            "py": data[5],
            "pz": data[6],
            "e": data[7],
        }
        for data in zip(x, y, z, cluster_e, px, py, pz, e, ovz)
        if data[8] <= 5000
    ]


def generate_data_interface(filename: str) -> List[DataInterface]:
    with uproot.open(filename) as file:
        tree: Dict[str, TBranch] = file["tuple/tuple;1"]
        data_interfaces = []
        data_extract = lambda d, keys: list(map(lambda k: d[k], keys))
        for i, (e_plus_data, e_minus_data) in enumerate(
            chunked(
                named_zip(
                    data,  # named tuple type
                    tree["EventNumber"].array(),
                    tree["ElectronTrack_PX"].array(),
                    tree["ElectronTrack_PY"].array(),
                    tree["ElectronTrack_PZ"].array(),
                    tree["electron_M"].array(),
                    tree["ElectronTrack_X"].array(),
                    tree["ElectronTrack_Y"].array(),
                    tree["ElectronTrack_Z"].array(),
                    tree["JPsi_PX"].array(),
                    tree["JPsi_PY"].array(),
                    tree["JPsi_PZ"].array(),
                    tree["JPsi_M"].array(),
                    tree["B_PX"].array(),
                    tree["B_PY"].array(),
                    tree["B_PZ"].array(),
                    tree["B_M"].array(),
                    tree["K_PX"].array(),
                    tree["K_PY"].array(),
                    tree["K_PZ"].array(),
                    tree["K_M"].array(),
                    tree["StdBremReco_Electron_PX"].array(),
                    tree["StdBremReco_Electron_PY"].array(),
                    tree["StdBremReco_Electron_PZ"].array(),
                    tree["StdBremReco_Electron_E"].array(),
                    tree["BremCluster_X"].array(),
                    tree["BremCluster_Y"].array(),
                    tree["BremCluster_Z"].array(),
                    tree["BremCluster_E"].array(),
                    tree["BremPhoton_PX"].array(),
                    tree["BremPhoton_PY"].array(),
                    tree["BremPhoton_PZ"].array(),
                    tree["BremPhoton_E"].array(),
                    tree["BremPhoton_OVZ"].array(),
                    tree["nBremPhotons"].array(),
                    tree["nElectronTracks"].array(),
                    tree["ElectronTrack_TYPE"].array(),
                )
            )
        ):
            # if data[0][-2] < 1 or data[0][-1][0] != 3 or data[1][-2] < 1 or data[0][-1][0] != 3:
            # not a well constructed track
            #    continue

            e_plus_data = e_plus_data._asdict()
            e_minus_data = e_minus_data._asdict()
            if (
                e_plus_data["nelectrons"] < 1  # some positive electron tracks
                or e_plus_data["etrack_type"][0]
                != 3  # the tracks are at least type 3 - hence well reconstructed
                or e_plus_data["nbrems"] < 1  # at least a single brem photon
                or e_minus_data["nelectrons"] < 1  # some negative electron tracks
                or e_minus_data["etrack_type"][0]
                != 3  # the tracks are at least type 3 - hence well reconstructed
                or e_minus_data["nbrems"] < 1  # at least a single brem photon
            ):
                continue
            # e_minus_data = data[1]
            brem_plus_data = generate_brem_data(
                *data_extract(
                    e_plus_data,
                    [
                        "brem_x",
                        "brem_y",
                        "brem_z",
                        "brem_cluster_e",
                        "brem_px",
                        "brem_py",
                        "brem_pz",
                        "brem_e",
                        "brem_ovz",
                    ],
                )
            )
            brem_minus_data = generate_brem_data(
                *data_extract(
                    e_minus_data,
                    [
                        "brem_x",
                        "brem_y",
                        "brem_z",
                        "brem_cluster_e",
                        "brem_px",
                        "brem_py",
                        "brem_pz",
                        "brem_e",
                        "brem_ovz",
                    ],
                )
            )
            if len(brem_plus_data) == 0 and len(brem_minus_data) == 0:
                continue

            data_dict = {
                "id": e_plus_data["event_number"],
                "e_plus_px": e_plus_data["e_px"][0],
                "e_plus_py": e_plus_data["e_py"][0],
                "e_plus_pz": e_plus_data["e_pz"][0],
                "e_plus_m": e_plus_data["e_m"],
                "e_plus_x": e_plus_data["e_x"][0],
                "e_plus_y": e_plus_data["e_y"][0],
                "e_plus_z": e_plus_data["e_z"][0],
                "e_minus_px": e_minus_data["e_px"][0],
                "e_minus_py": e_minus_data["e_py"][0],
                "e_minus_pz": e_minus_data["e_pz"][0],
                "e_minus_m": e_minus_data["e_m"],
                "e_minus_x": e_minus_data["e_x"][0],
                "e_minus_y": e_minus_data["e_y"][0],
                "e_minus_z": e_minus_data["e_z"][0],
                "jpsi_px": e_plus_data["j_px"],
                "jpsi_py": e_plus_data["j_py"],
                "jpsi_pz": e_plus_data["j_pz"],
                "jpsi_m": e_plus_data["j_m"],
                "b_px": e_plus_data["b_px"],
                "b_py": e_plus_data["b_py"],
                "b_pz": e_plus_data["b_pz"],
                "b_m": e_plus_data["b_m"],
                "k_px": e_plus_data["k_px"],
                "k_py": e_plus_data["k_py"],
                "k_pz": e_plus_data["k_pz"],
                "k_m": e_plus_data["k_m"],
                "std_electron_plus_px": e_plus_data["std_e_px"],
                "std_electron_plus_py": e_plus_data["std_e_py"],
                "std_electron_plus_pz": e_plus_data["std_e_pz"],
                "std_electron_plus_e": e_plus_data["std_e_e"],
                "std_electron_minus_px": e_minus_data["std_e_px"],
                "std_electron_minus_py": e_minus_data["std_e_py"],
                "std_electron_minus_pz": e_minus_data["std_e_pz"],
                "std_electron_minus_e": e_minus_data["std_e_e"],
                "brem_plus_data": brem_plus_data,
                "brem_minus_data": brem_minus_data,
            }
            d = DataInterface(deepcopy(data_dict))
            data_interfaces.append(d)
    return data_interfaces


C = 3 * (10 ** 8)


def reconstruct_brem(x: float, y: float, z: float, e: float, ov=None):
    if not isinstance(ov, list):
        ov = [0, 0, 0]
    xi = x - ov[0]
    yi = y - ov[0]
    zi = z - ov[0]
    mag = np.sqrt(xi ** 2 + yi ** 2 + zi ** 2)
    ratio = (1 / mag) * e / C
    return Momentum4(e, xi * ratio, yi * ratio, zi * ratio)


def find_closet_pairs(
    brem_cluster_e: List[float], brem_photon_e: List[float]
) -> List[int]:
    print(brem_photon_e)
    print(brem_cluster_e)
    print(len(brem_photon_e))
    print(len(brem_cluster_e))

    def closet_val_fn(iv: Tuple[int, float], indicies: List[float]):
        i, val = iv
        if i in indicies:
            return math.inf
        else:
            return val

    indicies = []
    for cluster_e in brem_cluster_e:
        while True:
            copied_brem_p = list(
                map(
                    partial(closet_val_fn, indicies=indicies),
                    enumerate(deepcopy(brem_photon_e)),
                )
            )
            print(copied_brem_p)
            idx = brem_photon_e.index(
                min(copied_brem_p, key=lambda x: abs(x - cluster_e))
            )
            if not idx in indicies:
                indicies.append(idx)
                break
    print(indicies)


def estimate_brem_momentum_variance(filename: str):
    sub_data_tuple = namedtuple(
        "sub_data_tuple",
        ["x", "y", "z", "c_e", "ovx", "ovy", "ovz", "px", "py", "pz", "t_e"],
    )
    true: List[Momentum4] = []
    reco_ov: List[Momentum4] = []
    reco: List[Momentum4] = []
    e = []
    with uproot.open(filename) as file:
        tree: Dict[str, TBranch] = file["tuple/tuple;1"]
        i = 0
        for data in zip(
            tree["BremCluster_X"].array(),
            tree["BremCluster_Y"].array(),
            tree["BremCluster_Z"].array(),
            tree["BremCluster_E"].array(),
            tree["BremPhoton_OVX"].array(),
            tree["BremPhoton_OVY"].array(),
            tree["BremPhoton_OVZ"].array(),
            tree["BremPhoton_PX"].array(),
            tree["BremPhoton_PY"].array(),
            tree["BremPhoton_PZ"].array(),
            tree["BremPhoton_E"].array(),
        ):
            if i == 0:
                find_closet_pairs(list(data[3]), list(data[-1]))
                i += 1
            for brem in named_zip(sub_data_tuple, *data):
                if brem.ovz > 5000:
                    continue
                e.append(brem.c_e)
                reco.append(reconstruct_brem(brem.x, brem.y, brem.z, brem.c_e))
                reco_ov.append(
                    reconstruct_brem(
                        brem.x,
                        brem.y,
                        brem.z,
                        brem.c_e,
                        ov=[brem.ovx, brem.ovy, brem.ovz],
                    )
                )
                true.append(Momentum4(brem.t_e, brem.px, brem.py, brem.pz))
    print(np.mean([t.p_x for t in true]))
    print(np.mean([t.p_y for t in true]))
    print(np.mean([t.p_z for t in true]))
    print(np.mean([t.p_x for t in reco]))
    print(np.mean([t.p_y for t in reco]))
    print(np.mean([t.p_z for t in reco]))
    print(np.mean([t.p_x for t in reco_ov]))
    print(np.mean([t.p_y for t in reco_ov]))
    print(np.mean([t.p_z for t in reco_ov]))
    print(np.mean([t.e for t in true]))
    print(np.mean([t.e for t in reco]))
    print(np.mean([t.e for t in reco_ov]))
    print(np.mean(e))
    print("--------")
    print(np.mean([np.abs(t.p_x - r.p_x) for t, r in zip(true, reco)]))
    print(np.mean([np.abs(t.p_y - r.p_y) for t, r in zip(true, reco)]))
    print(np.mean([np.abs(t.p_z - r.p_z) for t, r in zip(true, reco)]))
    print(np.mean([np.abs(t.e - r.e) for t, r in zip(true, reco)]))
    print(np.mean([np.abs(t.p_x - r.p_x) for t, r in zip(true, reco_ov)]))
    print(np.mean([np.abs(t.p_y - r.p_y) for t, r in zip(true, reco_ov)]))
    print(np.mean([np.abs(t.p_z - r.p_z) for t, r in zip(true, reco_ov)]))
    print(np.mean([np.abs(t.e - r.e) for t, r in zip(true, reco_ov)]))


def generate_data_mixing(
    data: List[DataInterface], sampling_frac: int = 2
) -> pd.DataFrame:
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
        length = sum(
            map(lambda x: 1, filter(lambda _internal_id: _internal_id == _id, brem_id))
        )
        false_mask = list(map(lambda _internal_id: _internal_id != _id, brem_id))
        sample_pos = sample(
            list(compress(brem_pos, false_mask)), k=int(sampling_frac * length / 2)
        )
        sample_momentum = sample(
            list(compress(brem_momentum, false_mask)), k=int(sampling_frac * length / 2)
        )
        d = data[i]
        mixed_data = d.generate_external_data_slice(sample_momentum, sample_pos)
        full.extend([deepcopy(group), mixed_data])
        i += 1
    return pd.concat(full)


def generate_prepared_data(
    data: pd.DataFrame, split_frac: int = 0.9, no_split: bool = False
):
    label_list = data["label"].to_numpy()
    new_df = data.drop(["label", "id", "e_energy", "b_energy"], axis=1)
    new_data = new_df.to_numpy()
    if no_split:
        return new_data, label_list
    else:
        indices = np.random.permutation(new_data.shape[0])
        i = int(split_frac * new_data.shape[0])
        training_idx, validation_idx = indices[:i], indices[i:]
        training_data, validation_data = (
            new_data[training_idx, :],
            new_data[validation_idx, :],
        )
        training_labels, validation_labels = (
            label_list[training_idx],
            label_list[validation_idx],
        )
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

    gbc = GradientBoostingClassifier(
        learning_rate=res.learning_rate[0], n_estimators=res.n_estimators[0]
    )
    gbc.fit(training_data, train_labels)
    return gbc


def train_xgboost(
    training_data: np.ndarray,
    training_labels: np.ndarray,
    validation_data: np.ndarray,
    validation_labels: np.ndarray,
):
    xgb = XGBClassifier(use_label_encoder=False)
    xgb.fit(training_data, training_labels, eval_metric="auc")
    train_acc = 100 * (
        sum(xgb.predict(training_data) == training_labels) / training_data.shape[0]
    )
    val_acc = 100 * (
        sum(xgb.predict(validation_data) == validation_labels)
        / validation_data.shape[0]
    )
    print(train_acc)
    print(val_acc)
    return xgb


def plot_xgboost_histo(
    xgb: XGBClassifier,
    training_data: np.ndarray,
    training_labels: np.ndarray,
    validation_data: np.ndarray,
    validation_labels: np.ndarray,
):
    train_acc = 100 * (
        sum(xgb.predict(training_data) == training_labels) / training_data.shape[0]
    )
    val_acc = 100 * (
        sum(xgb.predict(validation_data) == validation_labels)
        / validation_data.shape[0]
    )
    print(train_acc)
    print(val_acc)

    classifier_training_s = xgb.predict(
        training_data[training_labels == 1], output_margin=True
    )
    classifier_training_b = xgb.predict(
        training_data[training_labels == 0], output_margin=True
    )
    classifier_testing_s = xgb.predict(
        validation_data[validation_labels == 1], output_margin=True
    )
    classifier_testing_b = xgb.predict(
        validation_data[validation_labels == 0], output_margin=True
    )

    # classifier_training_s = xgb.predict_proba(training_data[training_labels==1])[:,1]
    # classifier_training_b = xgb.predict_proba(training_data[training_labels==0])[:,0]
    # #classifier_testing_s = xgb.predict_proba(validation_data)[:,1]
    # #classifier_testing_b = xgb.predict_proba(validation_data)[:,0]
    # print(xgb.predict_proba(training_data)[:5],training_labels[:10])

    c_min = -15
    c_max = 10

    histo_training_s = np.histogram(
        classifier_training_s, bins=80, range=(c_min, c_max), density=True
    )
    histo_training_b = np.histogram(
        classifier_training_b, bins=80, range=(c_min, c_max), density=True
    )
    histo_testing_s = np.histogram(
        classifier_testing_s, bins=80, range=(c_min, c_max), density=True
    )
    histo_testing_b = np.histogram(
        classifier_testing_b, bins=80, range=(c_min, c_max), density=True
    )

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

    # Figure size
    plt.figure(figsize=(16, 16))
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
    # plt.title("Classification with scikit-learn", fontsize = 18)
    plt.xlabel("Classifier, GBC", fontsize=20)
    plt.ylabel("Counts/Bin", fontsize=20)
    plt.xticks(size=18)
    plt.yticks(size=18)
    # Make legend with small font
    legend = ax1.legend(loc="upper center", shadow=True, ncol=2)
    for alabel in legend.get_texts():
        alabel.set_fontsize(22)

    plt.grid(alpha=0.5)
    plt.show()


def plot_roc(
    gbc: GradientBoostingClassifier,
    training_data,
    train_labels,
    validation_data,
    validation_labels,
):
    # fpr, tpr, thresholds = roc_curve(train_labels, gbc.decision_function(training_data))
    # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    roc_display_train = RocCurveDisplay.from_estimator(gbc, training_data, train_labels)
    roc_display_val = RocCurveDisplay.from_estimator(
        gbc, validation_data, validation_labels
    )
    roc_display_train.plot(ax=ax1, label="Train ")
    roc_display_val.plot(ax=ax2, label="Val")
    plt.show()


def plot_histo(
    gbc: GradientBoostingClassifier,
    training_data,
    train_labels,
    validation_data,
    validation_labels,
):
    train_acc = 100 * (
        sum(gbc.predict(training_data) == train_labels) / training_data.shape[0]
    )
    val_acc = 100 * (
        sum(gbc.predict(validation_data) == validation_labels)
        / validation_data.shape[0]
    )
    classifier_training_s = gbc.decision_function(
        training_data[train_labels == 1]
    ).ravel()
    classifier_training_b = gbc.decision_function(
        training_data[train_labels == 0]
    ).ravel()
    classifier_testing_s = gbc.decision_function(
        validation_data[validation_labels == 1]
    ).ravel()
    classifier_testing_b = gbc.decision_function(
        validation_data[validation_labels == 0]
    ).ravel()
    print(train_acc)
    print(val_acc)
    c_min = -10
    c_max = 10

    histo_training_s = np.histogram(
        classifier_training_s, bins=40, range=(c_min, c_max), density=True
    )
    histo_training_b = np.histogram(
        classifier_training_b, bins=40, range=(c_min, c_max), density=True
    )
    histo_testing_s = np.histogram(
        classifier_testing_s, bins=40, range=(c_min, c_max), density=True
    )
    histo_testing_b = np.histogram(
        classifier_testing_b, bins=40, range=(c_min, c_max), density=True
    )

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


def generate_hist_features(data: np.ndarray, range: Tuple[float, float]):
    histo = np.histogram(data, range=range, density=True)
    bin_edges = histo
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    return bin_widths, bin_centers


def plot_masses(
    classifier: XGBClassifier, data: List[DataInterface], cutoff: Optional[float] = None
):
    jpsi_noreco = []
    jpsi_truereco = []
    jpsi_stdreco = []
    jpsi_ourreco = []

    b_noreco = []
    b_truereco = []
    b_stdreco = []
    b_ourreco = []
    for d in data:
        jpsi_noreco.append(d.jpsi_noreco.m)
        jpsi_truereco.append(d.jpsi_truereco.m)
        jpsi_stdreco.append(d.jpsi_stdreco.m)
        jpsi_ourreco.append(d.jpsi_ourreco(classifier=classifier, cutoff=cutoff).m)

        b_noreco.append(d.b_noreco.m)
        b_truereco.append(d.b_truereco_from_electron.m)
        b_stdreco.append(d.b_stdreco.m)
        b_ourreco.append(d.b_ourreco(classifier=classifier, cutoff=cutoff).m)
    # pprint(b_ourreco)
    print(mode(jpsi_ourreco))
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 18))
    nbins = 30
    axes[0].hist(
        jpsi_noreco,
        label="no reco",
        histtype="stepfilled",
        range=(1000, 4000),
        bins=nbins,
        density=True,
        alpha=0.5,
    )
    axes[0].hist(
        jpsi_truereco,
        label="true reco",
        histtype="step",
        range=(1000, 4000),
        bins=nbins,
        density=True,
        color="black",
    )
    axes[0].hist(
        jpsi_stdreco,
        label="std reco",
        histtype="stepfilled",
        range=(1000, 4000),
        bins=nbins,
        density=True,
        alpha=0.5,
    )
    axes[0].hist(
        jpsi_ourreco,
        label="our reco",
        histtype="stepfilled",
        range=(1000, 6000),
        bins=nbins * 2,
        density=True,
        alpha=0.5,
    )

    axes[1].hist(
        b_noreco,
        label="no reco",
        histtype="stepfilled",
        range=(3000, 5600),
        bins=nbins,
        density=True,
        alpha=0.5,
    )
    axes[1].hist(
        b_truereco,
        label="true reco",
        histtype="step",
        range=(3000, 5600),
        bins=nbins,
        density=True,
        color="black",
    )
    axes[1].hist(
        b_stdreco,
        label="std reco",
        histtype="stepfilled",
        range=(3000, 5600),
        bins=nbins,
        density=True,
        alpha=0.5,
    )
    axes[1].hist(
        b_ourreco,
        label="our reco",
        histtype="stepfilled",
        range=(3000, 6000),
        bins=nbins * 2,
        density=True,
        alpha=0.5,
    )

    label_size = 16
    plt.rcParams["xtick.labelsize"] = label_size
    plt.rcParams["ytick.labelsize"] = label_size
    axes[0].set_ylabel("Counts/Bin", fontsize=18)
    axes[1].set_ylabel("Counts/Bin", fontsize=18)
    axes[0].grid(alpha=0.5)
    axes[1].grid(alpha=0.5)
    axes[0].set_xlim((1000, 6000))
    axes[1].set_xlim((3000, 6000))
    axes[0].legend(fontsize=24)
    axes[1].legend()
    axes[0].set_xlabel(r"$m_{J/psi}$ [MeV]", fontsize=18)
    axes[1].set_xlabel(r"$m_{B}$ [MeV]", fontsize=18)
    plt.show()


def eval_and_gen(
    filename: str, classifier: Optional[XGBClassifier] = None
) -> Optional[XGBClassifier]:
    data_interfaces = generate_data_interface(filename)
    data = generate_data_mixing(data_interfaces)
    (
        training_data,
        training_labels,
        validation_data,
        validation_labels,
    ) = generate_prepared_data(data)
    if classifier is not None:
        plot_masses(classifier, data_interfaces)
    else:
        classifier = train_xgboost(
            training_data, training_labels, validation_data, validation_labels
        )
        return classifier


def get_uncertainty_graphs(filename: str):
    with uproot.open(filename) as file:
        tree: Dict[str, TBranch] = file["tuple/tuple;1"]
        i = 0
        e_clusters = []
        true_clusters = []
        for (n_arr, e_arr, true_e_arr) in zip(
            tree["BremPhoton_nClusters"].array(),
            tree["BremCluster_E"].array(),
            tree["BremPhoton_E"].array(),
        ):
            prev_n = 0
            for n in n_arr:
                e_clusters.append(np.sum(e_arr.to_list()[prev_n : prev_n + int(n)]))
                prev_n += int(n)

            true_clusters.extend(true_e_arr)

        e_clusters = np.array(e_clusters)
        true_clusters = np.array(true_clusters)
        true_clusters = true_clusters[e_clusters != 0]
        e_clusters = e_clusters[e_clusters != 0]

        fig, axs = plt.subplots(2, 1, figsize=(18, 18))
        axs[0].hist(
            e_clusters / true_clusters,
            histtype="stepfilled",
            bins=50,
            density=True,
            range=(0, 100),
            color="blue",
        )
        axs[0].grid(alpha=0.5)
        axs[0].set_xlabel("Reconstructed Brem/True Brem", fontsize=18)
        axs[0].set_ylabel("Counts/Bin", fontsize=18)
        axs[0].set_xlim((0, 100))
        label_size = 16
        plt.rcParams["xtick.labelsize"] = label_size
        plt.rcParams["ytick.labelsize"] = label_size

        axs[1].plot(true_clusters, e_clusters / true_clusters, "bx")
        axs[1].set_xlim((0, 5000))
        axs[1].set_ylim((0, 1700))
        axs[1].set_xlabel("True Brem", fontsize=18)
        axs[1].set_ylabel("Reconstructed Brem/True Brem", fontsize=18)
        axs[1].grid(alpha=0.5)
        plt.show()


def plot_energy_disto(filename: str):
    with uproot.open(filename) as file:
        tree: Dict[str, TBranch] = file["tuple/tuple;1"]
        e_clusters = []
        true_clusters = []
        calo_clusters = []
        for (n_arr, e_arr, true_e_arr, cluster_e) in zip(
            tree["BremPhoton_nClusters"].array(),
            tree["BremCluster_E"].array(),
            tree["BremPhoton_E"].array(),
            tree["CaloCluster_E"].array(),
        ):
            prev_n = 0
            for n in n_arr:
                e_clusters.append(np.sum(e_arr.to_list()[prev_n : prev_n + int(n)]))
                prev_n += int(n)

            true_clusters.extend(true_e_arr)
            calo_clusters.extend(cluster_e)
        calo_clusters = np.array(calo_clusters)
        print(np.shape(calo_clusters))
        e_clusters = np.array(e_clusters)
        true_clusters = np.array(true_clusters)
        print(np.shape(true_clusters))
        true_clusters = true_clusters[e_clusters != 0]
        e_clusters = e_clusters[e_clusters != 0]
        calo_clusters = calo_clusters[calo_clusters != 0]
        print(calo_clusters.shape)
        print(true_clusters.shape)
        return e_clusters, true_clusters, calo_clusters


if __name__ == "__main__":
    data_interfaces = generate_data_interface("psiK_1000.root")
    # data_interfaces = generate_data_interface("Bu2JpsiK_ee_mu1.1_1000_events.root")
    data = generate_data_mixing(data_interfaces, sampling_frac=1)
    (
        training_data,
        training_labels,
        validation_data,
        validation_labels,
    ) = generate_prepared_data(data)

    xgb = train_xgboost(
        training_data, training_labels, validation_data, validation_labels,
    )

