from collections import namedtuple
from dataclasses import dataclass
import uproot
from uproot import TBranch
from typing import Dict, List, Optional, Tuple, Union
from utils import chunked, named_zip
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import compress
from random import sample
from copy import deepcopy
import vector
from xgboost import XGBClassifier, XGBRegressor
from vector import MomentumObject4D, VectorObject3D
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

data = namedtuple(
    "data",
    [
        "event_number",
        "e_px",
        "e_py",
        "e_pz",
        "e_m",
        "e_x",
        "e_y",
        "e_z",
        "j_px",
        "j_py",
        "j_pz",
        "j_m",
        "b_px",
        "b_py",
        "b_pz",
        "b_m",
        "k_px",
        "k_py",
        "k_pz",
        "k_m",
        "std_e_px",
        "std_e_py",
        "std_e_pz",
        "std_e_e",
        "brem_cluster_n_photons",
        "brem_x",
        "brem_y",
        "brem_z",
        "brem_cluster_e",
        "brem_px",
        "brem_py",
        "brem_pz",
        "brem_e",
        "brem_ovz",
        "nbrems",
        "nelectrons",
        "etrack_type",
    ],
)


def unpack_data(filename: str, cutoff: float = 5000):
    with uproot.open(filename) as file:
        tree: Dict[str, TBranch] = file["tuple/tuple;1"]
        datapoints: List[EventData] = []
        for (e_plus_data, e_minus_data) in chunked(
            named_zip(
                data,
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
                tree["BremPhoton_nClusters"].array(),
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
        ):
            if (
                e_plus_data.nelectrons < 1  # some positive electron tracks
                or e_plus_data.etrack_type[0]
                != 3  # the tracks are at least type 3 - hence well reconstructed
                or e_plus_data.nbrems < 1  # at least a single brem photon
                or e_minus_data.nelectrons < 1  # some negative electron tracks
                or e_minus_data.etrack_type[0]
                != 3  # the tracks are at least type 3 - hence well reconstructed
                or e_minus_data.nbrems < 1  # at least a single brem photon
            ):
                continue

            event_number = e_plus_data.event_number
            e_plus_brem_photons, e_plus_brem_positions = reco_brem_photons_nxyze(
                e_plus_data.brem_cluster_n_photons,
                e_plus_data.brem_x,
                e_plus_data.brem_y,
                e_plus_data.brem_z,
                e_plus_data.brem_cluster_e,
                cutoff=cutoff,
                ovz=e_plus_data.brem_ovz,
            )
            brem_plus_ovz = e_plus_data.brem_ovz
            brem_minus_ovz = e_minus_data.brem_ovz
            e_minus_brem_photons, e_minus_brem_positions = reco_brem_photons_nxyze(
                e_minus_data.brem_cluster_n_photons,
                e_minus_data.brem_x,
                e_minus_data.brem_y,
                e_minus_data.brem_z,
                e_minus_data.brem_cluster_e,
                cutoff=cutoff,
                ovz=e_minus_data.brem_ovz,
            )
            e_plus_true_photons = filter_by_ovz(
                e_plus_data.brem_ovz,
                reco_brem_photons_pxpypze(
                    e_plus_data.brem_px, e_plus_data.brem_py, e_plus_data.brem_pz, e_plus_data.brem_e
                ),
                cutoff,
            )
            e_minus_true_photons = filter_by_ovz(
                e_minus_data.brem_ovz,
                reco_brem_photons_pxpypze(
                    e_minus_data.brem_px, e_minus_data.brem_py, e_minus_data.brem_pz, e_minus_data.brem_e
                ),
                cutoff,
            )

            e_plus_std = vector.obj(
                px=e_plus_data.std_e_px,
                py=e_plus_data.std_e_py,
                pz=e_plus_data.std_e_pz,
                energy=e_plus_data.std_e_e,
            )

            e_minus_std = vector.obj(
                px=e_minus_data.std_e_px,
                py=e_minus_data.std_e_py,
                pz=e_minus_data.std_e_pz,
                energy=e_minus_data.std_e_e,
            )

            jpsi = vector.obj(
                px=e_plus_data.j_px, py=e_plus_data.j_py, pz=e_plus_data.j_pz, m=e_plus_data.j_m
            )
            k = vector.obj(px=e_plus_data.k_px, py=e_plus_data.k_py, pz=e_plus_data.k_pz, m=e_plus_data.k_m)
            b = vector.obj(px=e_plus_data.b_px, py=e_plus_data.b_py, pz=e_plus_data.b_pz, m=e_plus_data.b_m)

            e_plus = vector.obj(
                px=e_plus_data.e_px[0], py=e_plus_data.e_py[0], pz=e_plus_data.e_pz[0], m=e_plus_data.e_m
            )
            e_minus = vector.obj(
                px=e_minus_data.e_px[0], py=e_minus_data.e_py[0], pz=e_minus_data.e_pz[0], m=e_minus_data.e_m
            )

            e_plus_pos = vector.obj(x=e_plus_data.e_x[0], y=e_plus_data.e_y[0], z=e_plus_data.e_z[0])

            e_minus_pos = vector.obj(x=e_minus_data.e_x[0], y=e_minus_data.e_y[0], z=e_minus_data.e_z[0])

            datapoints.append(
                EventData(
                    event_number=event_number,
                    electron_plus_momentum=e_plus,
                    electron_plus_position=e_plus_pos,
                    electron_minus_momentum=e_minus,
                    electron_minus_position=e_minus_pos,
                    brem_plus_momenta=e_plus_brem_photons,
                    brem_plus_positions=e_plus_brem_positions,
                    true_brem_plus_momenta=e_plus_true_photons,
                    brem_plus_ovz=brem_plus_ovz,
                    brem_minus_momenta=e_minus_brem_photons,
                    brem_minus_positions=e_minus_brem_positions,
                    brem_minus_ovz=brem_minus_ovz,
                    true_brem_minus_momenta=e_minus_true_photons,
                    std_electron_plus_momentum=e_plus_std,
                    std_electron_minus_momentum=e_minus_std,
                    jpsi_momentum=jpsi,
                    k_momentum=k,
                    b_momentum=b,
                )
            )
    return datapoints


@dataclass
class EventData:
    event_number: int
    electron_plus_momentum: MomentumObject4D
    electron_plus_position: VectorObject3D
    electron_minus_momentum: MomentumObject4D
    electron_minus_position: VectorObject3D
    brem_plus_momenta: List[MomentumObject4D]
    true_brem_plus_momenta: List[MomentumObject4D]
    brem_plus_positions: List[VectorObject3D]
    brem_plus_ovz: List[float]
    brem_minus_momenta: List[MomentumObject4D]
    true_brem_minus_momenta: List[MomentumObject4D]
    brem_minus_positions: List[VectorObject3D]
    brem_minus_ovz: List[float]
    std_electron_minus_momentum: MomentumObject4D
    std_electron_plus_momentum: MomentumObject4D
    jpsi_momentum: MomentumObject4D
    b_momentum: MomentumObject4D
    k_momentum: MomentumObject4D

    def generate_data_slice(
        self,
        e_momentum: MomentumObject4D,
        e_pos: VectorObject3D,
        brem_momenta: List[MomentumObject4D],
        brem_positions: List[VectorObject3D],
        label: Optional[int] = None,
        cutoff: Optional[float] = None,
    ):
        if cutoff is None:
            cutoff = 0
        out = []
        for (brem_momentum, brem_pos) in zip(brem_momenta, brem_positions):
            if brem_momentum.E > cutoff:
                temp = {}
                dp: MomentumObject4D = e_momentum - brem_momentum
                dr: VectorObject3D = e_pos - brem_pos
                temp = {"p_dphi": dp.phi, "p_dtheta": dp.theta, "x_dtheta": dr.theta, "x_dphi": dr.phi}
                if label is not None:
                    temp.update({"id": self.event_number, "label": label})

                out.append(temp)

        return pd.DataFrame.from_records(out)

    def generate_external_data_slice(
        self, brem_momenta: List[MomentumObject4D], brem_positions: List[VectorObject3D]
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

    def full_brem_data(self) -> Tuple[List[MomentumObject4D], List[VectorObject3D], int]:
        brem_momentum = deepcopy(self.brem_minus_momenta)
        brem_momentum.extend(self.brem_plus_momenta)
        brem_pos = deepcopy(self.brem_minus_positions)
        brem_pos.extend(self.brem_plus_positions)
        return (brem_momentum, brem_pos, self.event_number)

    def full_brem_data_slice(self, threshold: float = 30, train: bool = True) -> pd.DataFrame:
        return pd.concat(
            [
                self.generate_brem_data_slice(
                    self.brem_minus_momenta,
                    self.true_brem_minus_momenta,
                    self.brem_minus_positions,
                    threshold,
                    train,
                ),
                self.generate_brem_data_slice(
                    self.brem_plus_momenta,
                    self.true_brem_plus_momenta,
                    self.brem_plus_positions,
                    threshold,
                    train,
                ),
            ]
        )

    def generate_brem_data_slice(
        self,
        reco: List[MomentumObject4D],
        true: List[MomentumObject4D],
        positions: List[VectorObject3D],
        threshold: float = 30,
        train: bool = True,
    ):
        out = []
        if not train:
            true = [0 for _ in range(len(reco))]
        for r, t, p in zip(reco, true, positions):
            temp = {
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "e": r.e,
                "eta": r.eta,
                "phi": r.phi,
                "pt": r.pt,
            }
            if train:
                temp["id"] = self.event_number
                if r.e / t.e > threshold:
                    temp["label"] = 0
                else:
                    temp["label"] = 1
            out.append(temp)
        return pd.DataFrame.from_records(out)

    @property
    def jpsi_noreco(self) -> MomentumObject4D:
        return self.electron_plus_momentum + self.electron_minus_momentum

    @property
    def jpsi_stdreco(self) -> MomentumObject4D:
        return self.std_electron_plus_momentum + self.std_electron_minus_momentum

    @property
    def jpsi_truereco(self) -> MomentumObject4D:
        eplus_true_reco = deepcopy(self.electron_plus_momentum)
        for brem_momentum in self.true_brem_plus_momenta:
            eplus_true_reco += brem_momentum
        eminus_true_reco = deepcopy(self.electron_minus_momentum)
        for brem_momentum in self.true_brem_minus_momenta:
            eminus_true_reco += brem_momentum
        return eplus_true_reco + eminus_true_reco

    @property
    def b_noreco(self) -> MomentumObject4D:
        return self.jpsi_noreco + self.k_momentum

    @property
    def b_stdreco(self) -> MomentumObject4D:
        return self.jpsi_stdreco + self.k_momentum

    @property
    def b_truereco_from_electron(self) -> MomentumObject4D:
        return self.jpsi_truereco + self.k_momentum

    @property
    def b_truereco(self) -> MomentumObject4D:
        return self.k_momentum + self.jpsi_momentum

    def method_0(self, e: MomentumObject4D, brem_arr: List[MomentumObject4D]):
        for brem in brem_arr:
            e += brem
        return e

    def method_1(self, e: MomentumObject4D, brem_arr: List[MomentumObject4D]):
        for brem in brem_arr:
            px = brem.px + e.px
            py = brem.py + e.py
            pz = brem.pz + e.pz
            energy = np.sqrt(e.m**2 + px**2 + py**2 + pz**2)
            e = vector.obj(e=energy, px=px, py=py, pz=pz)
        return e

    def method_2(self, e: MomentumObject4D, brem_arr: List[MomentumObject4D]):
        for brem in brem_arr:
            bp = brem.p
            ep = e.p
            p = bp + ep
            fac = p / ep
            px = e.px * fac
            py = e.py * fac
            pz = e.pz * fac
            energy = np.sqrt(e.m**2 + p**2)
            e = vector.obj(e=energy, px=px, py=py, pz=pz)
        return e

    def method_3(
        self,
        e: MomentumObject4D,
        brem_arr: List[MomentumObject4D],
        brem_pos: List[VectorObject3D],
        classifier: XGBClassifier,
    ):
        if len(brem_arr) > 0:
            brem_df = self.generate_brem_data_slice(brem_arr, None, brem_pos, train=False)
            brem_np = brem_df.to_numpy()
            pred = classifier.predict(brem_np) == 1
            for brem in compress(brem_arr, pred):
                e += brem
        return e

    def method_4(
        self,
        e: MomentumObject4D,
        brem_arr: List[MomentumObject4D],
    ):
        for brem in brem_arr:
            e += brem
        return e

    def jpsi_ourreco(
        self,
        classifier: XGBClassifier,
        cutoff: Optional[float] = None,
        m_method: int = 0,
        brem_classifier: Optional[XGBClassifier] = None,
    ) -> MomentumObject4D:
        full_brem_pos = deepcopy(self.brem_minus_positions)
        full_brem_momentum = deepcopy(self.brem_minus_momenta)
        full_brem_pos.extend(self.brem_plus_positions)
        full_brem_momentum.extend(self.brem_plus_momenta)
        plus_df = self.generate_data_slice(
            self.electron_plus_momentum,
            self.electron_plus_position,
            self.brem_plus_momenta,
            self.brem_plus_positions,
            cutoff=cutoff,
        )

        minus_df = self.generate_data_slice(
            self.electron_minus_momentum,
            self.electron_minus_position,
            self.brem_minus_momenta,
            self.brem_minus_positions,
            cutoff=cutoff,
        )
        plus_arr = plus_df.to_numpy()
        minus_arr = minus_df.to_numpy()
        e_plus_reco = deepcopy(self.electron_plus_momentum)
        e_minus_reco = self.electron_minus_momentum
        if len(plus_arr) > 0:
            plus_predictions: List[bool] = classifier.predict(plus_arr) == 1
            plus_brem_momentum: List[MomentumObject4D] = list(
                compress(self.brem_plus_momenta, plus_predictions)
            )
            if m_method == 0:
                e_plus_reco = self.method_0(e_plus_reco, plus_brem_momentum)
            elif m_method == 1:
                e_plus_reco = self.method_1(e_plus_reco, plus_brem_momentum)
            elif m_method == 2:
                e_plus_reco = self.method_2(e_plus_reco, plus_brem_momentum)
            elif m_method == 3 and brem_classifier is not None:
                plus_brem_positions: List[VectorObject3D] = list(
                    compress(self.brem_plus_positions, plus_predictions)
                )
                e_plus_reco = self.method_3(
                    e_plus_reco, plus_brem_momentum, plus_brem_positions, brem_classifier
                )
            elif m_method == 4:
                e_plus_reco = self.method_4(
                    e_plus_reco, list(compress(self.true_brem_plus_momenta, plus_predictions))
                )
            else:
                raise ValueError(f"not a valid method {m_method}")
        if len(minus_arr) > 0:
            minus_predictions: List[bool] = classifier.predict(minus_arr) == 1

            minus_brem_momentum: List[MomentumObject4D] = list(
                compress(self.brem_minus_momenta, minus_predictions)
            )
            if m_method == 0:
                e_minus_reco = self.method_0(e_minus_reco, minus_brem_momentum)
            elif m_method == 1:
                e_minus_reco = self.method_1(e_minus_reco, minus_brem_momentum)
            elif m_method == 2:
                e_minus_reco = self.method_2(e_minus_reco, minus_brem_momentum)
            elif m_method == 3 and brem_classifier is not None:
                minus_brem_positions: List[VectorObject3D] = list(
                    compress(self.brem_minus_positions, minus_predictions)
                )
                e_minus_reco = self.method_3(
                    e_minus_reco, minus_brem_momentum, minus_brem_positions, brem_classifier
                )
            elif m_method == 4:
                e_minus_reco = self.method_4(
                    e_minus_reco, list(compress(self.true_brem_minus_momenta, minus_predictions))
                )
            else:
                raise ValueError(f"not a valid method {m_method}")
        return e_plus_reco + e_minus_reco

    def b_ourreco(
        self,
        classifier: XGBClassifier,
        cutoff: Optional[float] = None,
        m_method: int = 0,
        brem_classifier: Optional[XGBClassifier] = None,
    ) -> MomentumObject4D:
        return (
            self.jpsi_ourreco(
                classifier=classifier, cutoff=cutoff, m_method=m_method, brem_classifier=brem_classifier
            )
            + self.k_momentum
        )


def reco_brem_photons_nxyze(n_arr, xi, yi, zi, ei, cutoff: float, ovz):
    brem_photons: List[MomentumObject4D] = []
    brem_positions: List[VectorObject3D] = []
    prev_n = 0
    for n in n_arr:
        energy = np.sum(ei.to_list()[prev_n : prev_n + int(n)])
        x_pos = np.mean(xi.to_list()[prev_n : prev_n + int(n)])
        y_pos = np.mean(yi.to_list()[prev_n : prev_n + int(n)])
        z_pos = np.mean(zi.to_list()[prev_n : prev_n + int(n)])
        brem = reco_brem_xyze(x_pos, y_pos, z_pos, energy)
        brem_photons.append(brem)
        brem_positions.append(vector.obj(x=x_pos, y=y_pos, z=z_pos))
        prev_n += int(n)

    return filter_by_ovz(ovz, brem_photons, cutoff), filter_by_ovz(ovz, brem_positions, cutoff)


def reco_brem_xyze(x: float, y: float, z: float, e: float):
    mag = np.sqrt(x**2 + y**2 + z**2)
    ratio = (1 / mag) * e
    return vector.obj(energy=e, px=x * ratio, py=y * ratio, pz=z * ratio)


def reco_brem_photons_pxpypze(px, py, pz, e):
    brem_photons: List[MomentumObject4D] = []
    for (pxi, pyi, pzi, ei) in zip(px, py, pz, e):
        brem_photons.append(vector.obj(px=pxi, py=pyi, pz=pzi, energy=ei))
    return brem_photons


def filter_by_ovz(
    ovz_arr: List[float], vec_arr: List[Union[MomentumObject4D, VectorObject3D]], cutoff: float = 5000
):
    return list(map(lambda x: x[1], filter(lambda x: x[0] <= 5000, zip(ovz_arr, vec_arr))))


def generate_data_mixing(data: List[EventData], sampling_frac: int = 1) -> pd.DataFrame:
    base_df = pd.concat([d.full_data_slice(label=1) for d in data])
    full = []
    brem_pos: List[VectorObject3D] = []
    brem_momentum: List[MomentumObject4D] = []
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


def generate_prepared_data(data: pd.DataFrame, split_frac: float = 0.9):
    label_list = data["label"].to_numpy()
    new_df = data.drop(["label", "id"], axis=1)
    new_data = new_df.to_numpy()
    indices = np.random.permutation(new_data.shape[0])
    i = int(split_frac * new_data.shape[0])
    training_idx, validation_idx = indices[:i], indices[i:]
    training_data, validation_data = new_data[training_idx, :], new_data[validation_idx, :]
    training_labels, validation_labels = label_list[training_idx], label_list[validation_idx]
    return training_data, training_labels, validation_data, validation_labels


def generate_brem_data_mixing(data: List[EventData], threshold: int = 10):
    """
    It takes a list of EventData objects, and for each EventData object, it takes the
    full_brem_data_slice, and then concatenates all of the slices together

    Args:
      data (List[EventData]): List[EventData]
      threshold (int): The threshold for the number of photons in a given event. Defaults to 10

    Returns:
      A dataframe with the same number of rows as the true dataframe, but with the same number of
    columns as the reco dataframe.
    """
    base_df = pd.concat([d.full_brem_data_slice(threshold=threshold) for d in data])

    true_df = base_df[base_df["label"] == 1]
    reco_df = base_df[base_df["label"] == 0].sample(true_df.shape[0], replace=True)
    print(reco_df.shape)
    return pd.concat([true_df, reco_df])


def train_brem_classifier(
    training_data: np.ndarray,
    training_labels: np.ndarray,
    validation_data: np.ndarray,
    validation_labels: np.ndarray,
):
    xgb = XGBClassifier()
    xgb.fit(training_data, training_labels)
    train_acc = 100 * (sum(xgb.predict(training_data) == training_labels) / training_data.shape[0])
    val_acc = 100 * (sum(xgb.predict(validation_data) == validation_labels) / validation_data.shape[0])
    print(train_acc)
    print(val_acc)
    return xgb


def test_brem_pre_label(filename: str):
    event_data = unpack_data(filename)
    df = generate_brem_data_mixing(event_data)
    training_data, training_labels, validation_data, validation_labels = generate_prepared_data(df)
    train_brem_classifier(training_data, training_labels, validation_data, validation_labels)


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


def train_xgboost(
    training_data: np.ndarray,
    training_labels: np.ndarray,
    validation_data: np.ndarray,
    validation_labels: np.ndarray,
):
    xgb = XGBClassifier()
    xgb.fit(training_data, training_labels)
    train_acc = 100 * (sum(xgb.predict(training_data) == training_labels) / training_data.shape[0])
    val_acc = 100 * (sum(xgb.predict(validation_data) == validation_labels) / validation_data.shape[0])
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
    train_acc = 100 * (sum(xgb.predict(training_data) == training_labels) / training_data.shape[0])
    val_acc = 100 * (sum(xgb.predict(validation_data) == validation_labels) / validation_data.shape[0])
    print(train_acc)
    print(val_acc)

    classifier_training_s = xgb.predict(training_data[training_labels == 1], output_margin=True)
    classifier_training_b = xgb.predict(training_data[training_labels == 0], output_margin=True)
    classifier_testing_s = xgb.predict(validation_data[validation_labels == 1], output_margin=True)
    classifier_testing_b = xgb.predict(validation_data[validation_labels == 0], output_margin=True)

    c_min = -10
    c_max = 10

    histo_training_s = np.histogram(classifier_training_s, bins=40, range=(c_min, c_max), density=True)
    histo_training_b = np.histogram(classifier_training_b, bins=40, range=(c_min, c_max), density=True)
    histo_testing_s = np.histogram(classifier_testing_s, bins=40, range=(c_min, c_max), density=True)
    histo_testing_b = np.histogram(classifier_testing_b, bins=40, range=(c_min, c_max), density=True)

    all_histos: List[Tuple[np.ndarray, np.ndarray]] = [
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


def eval_and_gen(filename: str, classifier: Optional[XGBClassifier] = None) -> Optional[XGBClassifier]:
    data_interfaces = unpack_data(filename)
    data = generate_data_mixing(data_interfaces)
    training_data, training_labels, validation_data, validation_labels = generate_prepared_data(data)
    if classifier is not None:
        plot_masses(classifier, data_interfaces)
    else:
        classifier = train_xgboost(training_data, training_labels, validation_data, validation_labels)
        return classifier


def plot_masses(
    classifier: XGBClassifier,
    data: List[EventData],
    cutoff: Optional[float] = None,
    m_method: int = 3,
    brem_classifier: Optional[XGBClassifier] = None,
):
    jpsi_noreco = []
    jpsi_truereco = []
    jpsi_stdreco = []
    jpsi_ourreco = []
    print(cutoff)
    b_noreco = []
    b_truereco = []
    b_stdreco = []
    b_ourreco = []
    for d in data:
        jpsi_noreco.append(d.jpsi_noreco.m)
        jpsi_truereco.append(d.jpsi_truereco.m)
        jpsi_stdreco.append(d.jpsi_stdreco.m)
        jpsi_ourreco.append(
            d.jpsi_ourreco(
                classifier=classifier, cutoff=cutoff, m_method=m_method, brem_classifier=brem_classifier
            ).m
        )

        b_noreco.append(d.b_noreco.m)
        b_truereco.append(d.b_truereco_from_electron.m)
        b_stdreco.append(d.b_stdreco.m)
        b_ourreco.append(
            d.b_ourreco(
                classifier=classifier, cutoff=cutoff, m_method=m_method, brem_classifier=brem_classifier
            ).m
        )
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

    label_size = 18

    plt.rcParams["xtick.labelsize"] = label_size

    plt.rcParams["ytick.labelsize"] = label_size

    axes[0].set_ylabel("Counts/Bin", fontsize=18)

    axes[1].set_ylabel("Counts/Bin", fontsize=18)

    axes[0].tick_params(axis="x", labelsize=14)
    axes[1].tick_params(axis="x", labelsize=14)

    axes[0].tick_params(axis="y", labelsize=14)
    axes[1].tick_params(axis="y", labelsize=14)

    axes[0].grid(alpha=0.5)

    axes[1].grid(alpha=0.5)

    axes[0].set_xlim((1000, 6000))

    axes[1].set_xlim((3000, 6000))

    axes[0].legend(prop={"size": 18})

    axes[1].legend(prop={"size": 18})

    axes[0].set_xlabel(r"$m_{J/psi}$ [MeV]", fontsize=18)

    axes[1].set_xlabel(r"$m_{B}$ [MeV]", fontsize=18)
    plt.savefig("plot_masses.svg")
    plt.show()
