from typing import List

from sklearn.multioutput import MultiOutputRegressor
from electron_reconstuction import EventData
from collections import namedtuple
from dataclasses import dataclass
import uproot
from tqdm import tqdm
from uproot import TBranch
from typing import Dict, List, Optional, Tuple, Union
from utils import chunked, named_zip
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain, compress, permutations
from random import sample
from copy import deepcopy
import vector
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from vector import MomentumObject4D, VectorObject3D
from sklearn.model_selection import train_test_split

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


def unpack_data(filename: str):
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
            )
            brem_plus_ovz = e_plus_data.brem_ovz
            brem_minus_ovz = e_minus_data.brem_ovz
            e_minus_brem_photons, e_minus_brem_positions = reco_brem_photons_nxyze(
                e_minus_data.brem_cluster_n_photons,
                e_minus_data.brem_x,
                e_minus_data.brem_y,
                e_minus_data.brem_z,
                e_minus_data.brem_cluster_e,
            )
            e_plus_true_photons = reco_brem_photons_pxpypze(
                e_plus_data.brem_px, e_plus_data.brem_py, e_plus_data.brem_pz, e_plus_data.brem_e
            )
            e_minus_true_photons = reco_brem_photons_pxpypze(
                e_minus_data.brem_px, e_minus_data.brem_py, e_minus_data.brem_pz, e_minus_data.brem_e
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


def reco_brem_photons_nxyze(n_arr, xi, yi, zi, ei):
    brem_photons: List[MomentumObject4D] = []
    brem_positions: List[VectorObject3D] = []
    prev_n = 0

    for n in filter(lambda x: x != 0, n_arr.to_list()):
        energy = np.sum(ei.to_list()[prev_n : prev_n + int(n)])
        x_pos = np.mean(xi.to_list()[prev_n : prev_n + int(n)])
        y_pos = np.mean(yi.to_list()[prev_n : prev_n + int(n)])
        z_pos = np.mean(zi.to_list()[prev_n : prev_n + int(n)])
        brem = reco_brem_xyze(x_pos, y_pos, z_pos, energy)
        brem_photons.append(brem)
        brem_positions.append(vector.obj(x=x_pos, y=y_pos, z=z_pos))
        prev_n += int(n)
    return brem_photons, brem_positions


def reco_brem_xyze(x: float, y: float, z: float, e: float):
    mag = np.sqrt(x**2 + y**2 + z**2)
    ratio = (1 / mag) * e
    return vector.obj(energy=e, px=x * ratio, py=y * ratio, pz=z * ratio)


def reco_brem_photons_pxpypze(px, py, pz, e):
    brem_photons: List[MomentumObject4D] = []
    for (pxi, pyi, pzi, ei) in zip(px, py, pz, e):
        brem_photons.append(vector.obj(px=pxi, py=pyi, pz=pzi, energy=ei))
    return brem_photons


def generate_electron_classification_training_data(
    data: List[EventData], filter_by_ovz: bool = False, split_frac: float = 0.9
):
    base: List[pd.DataFrame] = []
    for d in data:
        df = pd.concat(
            [
                electron_data_slice(
                    d.event_number,
                    d.electron_plus_momentum,
                    d.electron_plus_position,
                    d.brem_plus_momenta,
                    d.brem_plus_positions,
                    d.brem_plus_ovz,
                    filter_by_ovz,
                    label=1,
                ),
                electron_data_slice(
                    d.event_number,
                    d.electron_minus_momentum,
                    d.electron_minus_position,
                    d.brem_minus_momenta,
                    d.brem_minus_positions,
                    d.brem_minus_ovz,
                    filter_by_ovz,
                    label=1,
                ),
            ]
        )
        base.append(df)
    base_df: pd.DataFrame = pd.concat(base)
    brem_momentum: List[MomentumObject4D] = []
    brem_positions: List[VectorObject3D] = []
    brem_ovz: List[float] = []
    for d in data:
        brem_momentum.extend(d.brem_plus_momenta)
        brem_momentum.extend(d.brem_minus_momenta)
        brem_positions.extend(d.brem_plus_positions)
        brem_positions.extend(d.brem_minus_positions)
        brem_ovz.extend(d.brem_plus_ovz)
        brem_ovz.extend(d.brem_minus_ovz)

    brem_id = base_df["id"].to_list()
    i = 0

    full: List[pd.DataFrame] = []
    for _id, group in base_df.groupby("id"):
        length = group.shape[0]
        false_mask = list(map(lambda _internal_id: _internal_id != _id, brem_id))
        sample_positions = sample(list(compress(brem_positions, false_mask)), k=length)
        sample_momenta = sample(list(compress(brem_momentum, false_mask)), k=length)
        sample_ovz = sample(list(compress(brem_ovz, false_mask)), k=length)
        d = data[i]
        mixed_data = pd.concat(
            [
                electron_data_slice(
                    d.event_number,
                    d.electron_plus_momentum,
                    d.electron_plus_position,
                    sample_momenta,
                    sample_positions,
                    sample_ovz,
                    filter_by_ovz,
                    label=0,
                ),
                electron_data_slice(
                    d.event_number,
                    d.electron_minus_momentum,
                    d.electron_minus_position,
                    sample_momenta,
                    sample_positions,
                    sample_ovz,
                    filter_by_ovz,
                    label=0,
                ),
            ]
        )
        full.append(mixed_data)
        i += 1
    full_df = pd.concat(full)
    combined_df = pd.concat([base_df, full_df])
    shuffled_df = combined_df.sample(frac=1)
    y = shuffled_df["label"].to_numpy()
    X = shuffled_df.drop(["label", "id"], axis=1).to_numpy()
    training_data, validation_data, training_labels, validation_labels = train_test_split(
        X, y, test_size=split_frac
    )
    return training_data, training_labels, validation_data, validation_labels


def train_classifier(
    training_data: np.ndarray,
    training_labels: np.ndarray,
    validation_data: np.ndarray,
    validation_labels: np.ndarray,
    name: str = "Default",
):
    classifier = XGBClassifier()
    classifier.fit(training_data, training_labels)
    print(f"{name} Training Accuracy: {accuracy_score(training_labels, classifier.predict(training_data))}")
    if len(validation_data) > 0:
        print(
            f"{name} Validation Accuracy: {accuracy_score(validation_labels, classifier.predict(validation_data))}"
        )
    return classifier


def train_regressor(
    training_data: np.ndarray,
    training_output: np.ndarray,
    validation_data: np.ndarray,
    validation_output: np.ndarray,
    name: str = "Default",
):
    regressor = XGBRegressor()
    regressor.fit(training_data, training_output)
    print(f"{name} Training MSE: {mean_squared_error(training_output, regressor.predict(training_data))}")
    if len(validation_data) > 0:
        print(
            f"{name} Validation MSE: {mean_squared_error(validation_output, regressor.predict(validation_data))}"
        )
    return regressor


def train_multidimensional_regressor(
    training_data: np.ndarray,
    training_output: np.ndarray,
    validation_data: np.ndarray,
    validation_output: np.ndarray,
    name: str = "Default",
):
    regressor = MultiOutputRegressor(XGBRegressor())
    regressor.fit(training_data, training_output)
    print(f"{name} Training MSE: {mean_squared_error(training_output, regressor.predict(training_data))}")
    if len(validation_data) > 0:
        print(
            f"{name} Validation MSE: {mean_squared_error(validation_output, regressor.predict(validation_data))}"
        )
    return regressor


def electron_data_slice(
    event_number: int,
    electron_momentum: MomentumObject4D,
    electron_position: VectorObject3D,
    brem_momenta: List[MomentumObject4D],
    brem_positions: List[VectorObject3D],
    brem_ovz: Optional[List[float]],
    filter_by_ovz: bool,
    label: int = 1,
):
    """
    Given an electron momentum, electron position, brem momenta, brem positions, and brem ovz,
    return a pandas dataframe with the following columns:

    p_dphi, p_dtheta, x_dtheta, x_dphi, id, label

    The brem ovz is optional. If it is not provided, it will be assumed to be 0.

    If the label is not provided, defaults to 1 - this corresponds to a true label

    The id is the event number

    The function will filter out brems with ovz > 5000 if filter_by_ovz is set to True.

    Args:
      event_number (int): The event number
      electron_momentum (MomentumObject4D): The momentum of the electron.
      electron_position (VectorObject3D): The position of the electron
      brem_momenta (List[MomentumObject4D]): List[MomentumObject4D]
      brem_positions (List[VectorObject3D]): List[VectorObject3D]
      brem_ovz (Optional[List[float]]): If you want to filter out brem photons that are outside of the
    acceptance of the detector, set this to a list of the origin z of the brem photons.
      filter_by_ovz (bool): If True, only include brem photons with an origin in the active volume.
      label (int): int = 1,. Defaults to 1

    Returns:
      A pandas dataframe with the following columns:
        p_dphi: The phi angle between the electron momentum and the brem momentum
        p_dtheta: The theta angle between the electron momentum and the brem momentum
        x_dtheta: The theta angle between the electron position and the brem position
        x_dphi: The phi angle between the electron position
        id: the event number id
        label: the label, either 1 or 0
    """
    if brem_ovz is None:
        brem_ovz = [0.0] * len(brem_momenta)
    out = []
    for (brem_momentum, brem_pos, ovz) in zip(brem_momenta, brem_positions, brem_ovz):
        if ovz > 5000 and filter_by_ovz:
            continue
        temp = {}
        dp: MomentumObject4D = electron_momentum - brem_momentum
        dr: VectorObject3D = electron_position - brem_pos
        temp = {
            "p_dphi": dp.phi,
            "p_dtheta": dp.theta,
            "x_dtheta": dr.theta,
            "x_dphi": dr.phi,
            "id": event_number,
            "label": label,
        }
        out.append(temp)
    return pd.DataFrame.from_records(out)


def train_electron_classifier(
    event_data: Optional[List[EventData]], filename: str, pre_filter_ovz: bool, split_frac: float = 0.9
):
    if event_data is None:
        event_data: List[EventData] = unpack_data(filename)
    (
        training_data,
        training_labels,
        validation_data,
        validation_labels,
    ) = generate_electron_classification_training_data(event_data, pre_filter_ovz, split_frac)
    classifier = train_classifier(
        training_data, training_labels, validation_data, validation_labels, name="Electron Classifier"
    )
    return event_data, classifier


def generate_brem_regressor_data(event_data: List[EventData], filter_by_ovz: bool, split_frac: float = 0.9):
    """
    It takes the list of EventData objects and for each EventData object, parses the brem photon momentum and positions.
    It then generates training data based on the features of the reconstructed brem photon with an output equal to the ratio
    of the energy of the true brem photon to the reconstructed brem photon. This will then be used to train a regressor
    to predict this ratio. It then takes this dataset and splits it into training and validation sets

    Args:
      event_data (List[EventData]): List[EventData]
      filter_by_ovz (bool): bool = True
      split_frac (float): float = 0.9

    Returns:
      The training data, training labels, validation data, and validation labels.
    """
    brem_momenta: List[MomentumObject4D] = []
    brem_positions: List[VectorObject3D] = []
    true_brem_momenta: List[MomentumObject4D] = []
    brem_ovz = []
    for d in event_data:
        brem_momenta.extend(d.brem_plus_momenta)
        brem_momenta.extend(d.brem_minus_momenta)
        brem_positions.extend(d.brem_plus_positions)
        brem_positions.extend(d.brem_minus_positions)
        true_brem_momenta.extend(d.true_brem_plus_momenta)
        true_brem_momenta.extend(d.true_brem_plus_momenta)
        brem_ovz.extend(d.brem_plus_ovz)
        brem_ovz.extend(d.brem_minus_ovz)
    if filter_by_ovz:
        boolean_mask: List[bool] = list(map(lambda ovz: ovz <= 5000, brem_ovz))
        brem_momenta = list(compress(brem_momenta, boolean_mask))
        brem_positions = list(compress(brem_positions, boolean_mask))
        true_brem_momenta = list(compress(true_brem_momenta, boolean_mask))
    X = []
    y = []
    for p, t, x in zip(brem_momenta, true_brem_momenta, brem_positions):
        data_slice = brem_regressor_data_slice(p, x)
        if not any(np.isnan(data_slice)):
            X.append(data_slice)
            y.append(p.e / t.e)
    X = np.array(X)
    y = np.array(y)
    training_data, validation_data, training_labels, validation_labels = train_test_split(
        X, y, test_size=split_frac
    )

    return training_data, training_labels, validation_data, validation_labels


def brem_regressor_data_slice(brem_momenta: MomentumObject4D, brem_position: VectorObject3D):
    return [
        brem_momenta.eta,
        brem_momenta.phi,
        brem_momenta.pt,
        brem_momenta.e,
        brem_position.x,
        brem_position.y,
        brem_position.z,
    ]


def brem_classifier_data_slice(brem_momenta: MomentumObject4D, brem_position: VectorObject3D):
    return [
        brem_momenta.eta,
        brem_momenta.phi,
        brem_momenta.pt,
        brem_momenta.e,
        brem_position.x,
        brem_position.y,
        brem_position.z,
    ]


def generate_brem_classifier_data(
    event_data: List[EventData], filter_by_ovz: bool, split_frac: float = 0.9, threshold: float = 10
):
    """
    It takes in a list of EventData objects, and for each EventData object, parses brem photon momentum and positions. It then uses the ratio of the
    true brem photon energy to the reconstructed brem photon energy to generate a training dataset with labels 1 and 0, to
    train a classifier to be able to recognise brem photons with a ratio greater than particular cutoff.
    It then splits the data into training and validation sets

    Args:
      event_data (List[EventData]): List[EventData]
      filter_by_ovz (bool): If True, only keep events with an OVZ of less than 5000.
      split_frac (float): float = 0.9
      threshold (float): float = 10. Defaults to 10

    Returns:
      The training data, training labels, validation data, and validation labels.
    """
    brem_momenta: List[MomentumObject4D] = []
    brem_positions: List[VectorObject3D] = []
    true_brem_momenta: List[MomentumObject4D] = []
    brem_ovz = []
    for d in event_data:
        brem_momenta.extend(d.brem_plus_momenta)
        brem_momenta.extend(d.brem_minus_momenta)
        brem_positions.extend(d.brem_plus_positions)
        brem_positions.extend(d.brem_minus_positions)
        true_brem_momenta.extend(d.true_brem_plus_momenta)
        true_brem_momenta.extend(d.true_brem_plus_momenta)
        brem_ovz.extend(d.brem_plus_ovz)
        brem_ovz.extend(d.brem_minus_ovz)
    if filter_by_ovz:
        boolean_mask: List[bool] = list(map(lambda ovz: ovz <= 5000, brem_ovz))
        brem_momenta = list(compress(brem_momenta, boolean_mask))
        brem_positions = list(compress(brem_positions, boolean_mask))
        true_brem_momenta = list(compress(true_brem_momenta, boolean_mask))
    X = []
    y = []
    for p, t, x in zip(brem_momenta, true_brem_momenta, brem_positions):
        data_slice = brem_classifier_data_slice(p, x)
        if not any(np.isnan(data_slice)):
            if p.e / t.e > threshold:
                y.append(0)
            else:
                y.append(1)
            X.append(data_slice)
    X = np.array(X)
    y = np.array(y)
    training_data, validation_data, training_labels, validation_labels = train_test_split(
        X, y, train_size=split_frac
    )
    return training_data, training_labels, validation_data, validation_labels


def train_brem_ratio_regressor(
    event_data: Optional[List[EventData]], filename: str, pre_filter_ovz: bool, split_frac: float = 0.9
):
    if event_data is None:
        event_data = unpack_data(filename)
    (
        training_data,
        training_labels,
        validation_data,
        validation_labels,
    ) = generate_brem_regressor_data(event_data, pre_filter_ovz, split_frac)
    regressor = train_regressor(
        training_data,
        training_labels,
        validation_data,
        validation_labels,
        name="Bremsstrahlung Ratio Regressor",
    )
    return event_data, regressor


def train_brem_ratio_classifier(
    event_data: Optional[List[EventData]],
    filename: str,
    pre_filter_ovz: bool,
    split_frac: float = 0.9,
    threshold: float = 20,
):
    if event_data is None:
        event_data = unpack_data(filename)
    (
        training_data,
        training_labels,
        validation_data,
        validation_labels,
    ) = generate_brem_classifier_data(event_data, pre_filter_ovz, split_frac, threshold)
    classifier = train_classifier(
        training_data,
        training_labels,
        validation_data,
        validation_labels,
        name="Bremsstrahlung Ratio Classifier",
    )
    return event_data, classifier


def train_brem_ratio_multiclassifier(
    event_data: Optional[List[EventData]],
    filename: str,
    pre_filter_ovz: bool = True,
    split_frac: float = 0.9,
    thresholds: List[float] = [5, 10, 20],
):
    if event_data is None:
        event_data = unpack_data(filename)
    (
        training_data,
        training_labels,
        validation_data,
        validation_labels,
    ) = generate_brem_classifier_multiclassifier(event_data, pre_filter_ovz, split_frac, thresholds)
    classifier = train_classifier(training_data, training_labels, validation_data, validation_labels)
    return event_data, classifier


def generate_brem_classifier_multiclassifier(
    event_data: List[EventData], filter_by_ovz: bool, split_frac: float, thresholds: List[float]
):
    brem_momenta: List[MomentumObject4D] = []
    brem_positions: List[VectorObject3D] = []
    true_brem_momenta: List[MomentumObject4D] = []
    brem_ovz = []
    for d in event_data:
        brem_momenta.extend(d.brem_plus_momenta)
        brem_momenta.extend(d.brem_minus_momenta)
        brem_positions.extend(d.brem_plus_positions)
        brem_positions.extend(d.brem_minus_positions)
        true_brem_momenta.extend(d.true_brem_plus_momenta)
        true_brem_momenta.extend(d.true_brem_plus_momenta)
        brem_ovz.extend(d.brem_plus_ovz)
        brem_ovz.extend(d.brem_minus_ovz)
    if filter_by_ovz:
        boolean_mask: List[bool] = list(map(lambda ovz: ovz <= 5000, brem_ovz))
        brem_momenta = list(compress(brem_momenta, boolean_mask))
        brem_positions = list(compress(brem_positions, boolean_mask))
        true_brem_momenta = list(compress(true_brem_momenta, boolean_mask))
    X = []
    y = []
    thresholds.insert(0, 0)
    max_i = len(thresholds) - 1
    for p, t, x in zip(brem_momenta, true_brem_momenta, brem_positions):
        data_slice = brem_classifier_data_slice(p, x)
        if not any(np.isnan(data_slice)):
            thresholded = False
            for i, (lower, upper) in enumerate(zip(thresholds[:-1], thresholds[1:])):
                if upper > p.e / t.e > lower:
                    y.append(i)
                    thresholded = True
            if not thresholded:
                y.append(max_i)
            X.append(data_slice)
    X = np.array(X)
    y = np.array(y)
    training_data, validation_data, training_labels, validation_labels = train_test_split(
        X, y, train_size=split_frac
    )
    return training_data, training_labels, validation_data, validation_labels


def generate_electron_regressor_data(
    event_data: List[EventData], filter_by_ovz: bool, split_frac: float = 0.9
):
    base = []
    for d in tqdm(event_data):
        base.append(
            pd.concat(
                [
                    electron_regressor_data_slice(
                        d.electron_plus_momentum,
                        d.true_brem_plus_momenta,
                        d.brem_plus_positions,
                        d.brem_plus_momenta,
                        d.brem_plus_ovz,
                        filter_by_ovz,
                    ),
                    electron_regressor_data_slice(
                        d.electron_minus_momentum,
                        d.true_brem_minus_momenta,
                        d.brem_minus_positions,
                        d.brem_minus_momenta,
                        d.brem_minus_ovz,
                        filter_by_ovz,
                    ),
                ]
            )
        )
    base_df = pd.concat(base)
    target_labels = ["target_px", "target_py", "target_pz", "target_e"]
    y = base_df[target_labels].to_numpy()
    X = base_df.drop(target_labels).to_numpy()
    training_data, validation_data, training_labels, validation_labels = train_test_split(
        X, y, test_size=split_frac
    )
    return training_data, training_labels, validation_data, validation_labels


def electron_regressor_data_slice(
    electron_momentum: MomentumObject4D,
    true_brem_momenta: List[MomentumObject4D],
    brem_positions: List[VectorObject3D],
    brem_momenta: List[MomentumObject4D],
    brem_ovz: List[float],
    filter_by_ovz: bool,
):
    out = []
    if filter_by_ovz:
        boolean_mask = list(map(lambda ovz: ovz <= 5000, brem_ovz))
        brem_momenta = list(compress(brem_momenta, boolean_mask))
        brem_positions = list(compress(brem_positions, boolean_mask))
        true_brem_momenta = list(compress(true_brem_momenta, boolean_mask))
    for i in range(1, len(true_brem_momenta) + 1):
        for perm in permutations(range(len(true_brem_momenta)), r=i):
            end_idx = perm[-1]
            true_momenta = [true_brem_momenta[i] for i in perm]
            brem_momentum = brem_momenta[end_idx]
            brem_position = brem_positions[end_idx]
            e_target = deepcopy(electron_momentum)
            e_base = deepcopy(electron_momentum)
            for t in true_momenta:
                e_target += t
            for t in true_momenta[:-1]:
                e_base += t
            temp = {
                "electron_px": e_base.px,
                "electron_py": e_base.py,
                "electron_py": e_base.pz,
                "electron_e": e_base.e,
                "brem_px": brem_momentum.px,
                "brem_py": brem_momentum.py,
                "brem_pz": brem_momentum.pz,
                "brem_e": brem_momentum.e,
                "brem_x": brem_position.x,
                "brem_y": brem_position.y,
                "brem_z": brem_position.z,
                "target_px": e_target.px,
                "target_py": e_target.py,
                "target_pz": e_target.pz,
                "target_e": e_target.e,
            }
            out.append(temp)
    return pd.DataFrame.from_records(out)


def train_electron_reco_regressor(
    event_data: Optional[List[EventData]], filename: str, pre_filter_ovz: bool, split_frac: float = 0.9
):
    if event_data is None:
        event_data = unpack_data(filename)
    (
        training_data,
        training_labels,
        validation_data,
        validation_labels,
    ) = generate_electron_regressor_data(event_data, pre_filter_ovz, split_frac)
    print("Generated data for electron reco regressor")
    classifier = train_multidimensional_regressor(
        training_data, training_labels, validation_data, validation_labels
    )
    return event_data, classifier


if __name__ == "__main__":
    ...
