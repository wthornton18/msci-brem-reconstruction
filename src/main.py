from collections import defaultdict
from re import sub
from typing import Dict, List
import numpy
import uproot
from uproot import TTree, TBranch
import matplotlib.pyplot as plt
import itertools
from pprint import pprint
import math
from copy import deepcopy
import pandas as pd
import tensorflow.keras as keras



class InitialModel:
    def __init__(self, filename) -> None:
        self.filename = filename

    def machine_learning_model(self):
        model = keras.models.Sequential(
            [keras.layers.Flatten(input_shape=(10,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(2)
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()]
        )

        return model

    def prepare_data(self, df: pd.DataFrame, split_frac = 0.9):
        label_list = df['label'].to_numpy()
        new_df = df.drop(['label', 'id'], axis=1)
        new_data = new_df.to_numpy()
        indices = numpy.random.permutation(new_data.shape[0])
        i = int(split_frac * new_data.shape[0])
        training_idx, validation_idx = indices[:i], indices[i:]
        training_data, validation_data = new_data[training_idx,:], new_data[validation_idx,:]
        training_labels, validation_labels = label_list[training_idx], label_list[validation_idx]
        return training_data, training_labels, validation_data, validation_labels
    

        

    def train_model(self, model: keras.models.Sequential, training_data, train_labels, validation_data, validation_labels):
        model.fit(
            x=training_data,
            y=train_labels,
            epochs=6,
            validation_data=(validation_data, validation_labels)
        )

    def get_names(self):
        with uproot.open(self.filename) as file:
            tree: Dict[str, TBranch] = file['tuple/tuple;1']
            print(tree.keys())

    def generate_data_mapping(self):
        
        with uproot.open(self.filename) as file:
            tree: Dict[str, TBranch] = file['tuple/tuple;1']
            electron_data_sets = []
            for data in zip(tree['ElectronTrack_PX'].array(), 
                            tree['ElectronTrack_PY'].array(), 
                            tree['ElectronTrack_PZ'].array(),
                            tree['ElectronTrack_X'].array(),
                            tree['ElectronTrack_Y'].array(),
                            tree['ElectronTrack_Z'].array(),
                            tree['BremCluster_E'].array(),
                            tree['BremCluster_X'].array(),
                            tree['BremCluster_Y'].array(),
                            tree['BremCluster_Z'].array()):

                boolean_filter =  list(map(lambda x: x[0] and x[1] and x[2], 
                                    zip(map(math.isfinite, data[0]), 
                                    map(math.isfinite, data[1]), 
                                    map(math.isfinite, data[2]))))
                first_filtering = {
                    'ElectronTrack_PX': list(filter(math.isfinite, data[0])),
                    'ElectronTrack_PY': list(filter(math.isfinite, data[1])),
                    'ElectronTrack_PZ': list(filter(math.isfinite, data[2]))
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
                    {'ElectronTrack_X': list(itertools.compress(data[3], boolean_filter))[0],
                    'ElectronTrack_Y': list(itertools.compress(data[4], boolean_filter))[0],
                    'ElectronTrack_Z': list(itertools.compress(data[5], boolean_filter))[0],
                    'BremCluster_E': list(data[6]),
                    'BremCluster_X': list(data[7]),
                    'BremCluster_Y': list(data[8]),
                    'BremCluster_Z': list(data[9])}
                )
                electron_data_sets.append(deepcopy(second_filtering))
        switched_electron_data = []
        for i, data in enumerate(electron_data_sets):
            temp = {'id': i}
            temp.update({k:v if "ElectronTrack" in k else None for k, v in data.items()})
            brem_e, brem_x, brem_y, brem_z = data['BremCluster_E'], data['BremCluster_X'], data['BremCluster_Y'], data['BremCluster_Z']
            for brem_data in zip(brem_e, brem_x, brem_y, brem_z):
                refactored_data = deepcopy(temp)
                refactored_data.update({
                    'BremCluster_E': brem_data[0],
                    'BremCluster_X': brem_data[1],
                    'BremCluster_Y': brem_data[2],
                    'BremCluster_Z': brem_data[3]
                    })
                switched_electron_data.append(refactored_data)

        df = pd.DataFrame.from_dict(switched_electron_data)
        return df

    def generate_data_mixing(self, df: pd.DataFrame):
        ids = df['id'].unique().tolist()
        mixed_data = []
        columns_to_replace = [
            'ElectronTrack_PX',
            'ElectronTrack_PY', 
            'ElectronTrack_PZ',
            'ElectronTrack_X',
            'ElectronTrack_Y',
            'ElectronTrack_Z']
        for id in ids[:1]:
            running_df = df[df['id']==id]
            running_df['label'] = 1
            sampled_df = df[df['id']!=id].sample(len(running_df.index.tolist()))
            for column in columns_to_replace:
                sampled_df[column] = running_df[column].head(1).to_list()[0]
            sampled_df['label'] = 0
            combined_df = pd.concat([running_df, sampled_df])
            mixed_data.append(combined_df)

        return pd.concat(mixed_data)

im = InitialModel('1000ev.root')
#im.get_names()
df = im.generate_data_mapping()
mixed_data_groups = im.generate_data_mixing(df)
training_data, training_labels, validation_data, validation_labels = im.prepare_data(mixed_data_groups)
model = im.machine_learning_model()
im.train_model(model, training_data, training_labels, validation_data, validation_labels)