
import uproot
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import awkward as ak
from copy import deepcopy
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
#from pydotplus import graph_from_dot_data
from IPython.display import Image
#from sklearn.tree import export_graphviz
from sklearn.ensemble import GradientBoostingClassifier
#import xgboost as xgb
#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.options.mode.chained_assignment = None
plt.rcParams['axes.xmargin'] = 0
#pd.set_option('display.max_rows', 400)

class DataPrep:
    def __init__(self, filename, features, N = 1000) -> None:
        self.filename = filename
        self.features = features
        self.N = N  # number of events to access in the file

    def extract_data(self):
        with uproot.open(self.filename) as file:
            tree = file['tuple/tuple']
            Cluster_info = tree.arrays(self.features[6:], library = 'pd')
            ElectronTrack = tree.arrays(self.features[:6], library = 'pd')
            nElectronTracks = tree['nElectronTracks'].array()
            nCaloClusters = tree['nCaloClusters'].array()
            ElectronTrack_TYPE = tree['ElectronTrack_TYPE'].array()
            
            events = np.arange(0,self.N,1)
            nET = []  # nET = no Electron Tracks, i.e. remove events with no track reconstruction
            for i in tqdm(events):
                if nElectronTracks[i]< 1:
                    nET.append(i)
                elif ElectronTrack_TYPE[i,0] !=3:
                    nET.append(i)
            nET = np.array(nET)

            

            #Remove rows with no ElectronTracks
            mask1_indices = np.isin(events,nET)
            events_after_mask1 = events[~mask1_indices]
            print(events_after_mask1.shape)

            #Further Remove rows with no BremClusters
            nCI = []  # nCI = no Cluster Info, i.e. remove events with no cluster info
            for i in tqdm(events_after_mask1):
                if nCaloClusters[i] == 0:
                    nCI.append(i)
            nCI = np.array(nCI)
            
            mask2_indices = np.isin(events_after_mask1,nCI)
            events_after_mask2 = events_after_mask1[~mask2_indices]
            print(events_after_mask2.shape)

            # Extract the info from tree with correct indices
            ET = ElectronTrack.loc[events_after_mask2, self.features[:6]]
            CI = Cluster_info.loc[events_after_mask2, self.features[6:]]   
            
            # Remove nan's and inf's
            ET = ET[~ET.isin([np.nan, np.inf, -np.inf]).any(1)]
            CI = CI[~CI.isin([np.nan, np.inf, -np.inf]).any(1)]

            #Select only subentries with val 0 in pandas correspondong to TrackType 3 [because of first for loop]
            idx = pd.IndexSlice
            ET = ET.loc[idx[:,0],:]

            # Merging even and odd rows seperatly
            merge_BC_ET = pd.concat([ET, CI], axis=1).fillna(method='ffill')

            merge_BC_ET['label'] = 1
            df = merge_BC_ET.sort_index()

        return df.droplevel(level = 'subentry').reset_index().rename(columns = {'entry':'id'})

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
        for id in tqdm(ids[:10000]):
            running_df = df[df['id']==id]
            running_df['label'] = 1
            sampled_df = df[df['id']!=id].sample(len(running_df.index.tolist()))
            for column in columns_to_replace:
                sampled_df[column] = running_df[column].head(1).to_list()[0]
            sampled_df['label'] = 0
            combined_df = pd.concat([running_df, sampled_df])
            mixed_data.append(combined_df)

        return pd.concat(mixed_data)

    def prepare_data(self, df: pd.DataFrame, split_frac = 0.9):
        label_list = df['label'].to_numpy()
        new_df = df.drop(['label', 'id'], axis=1)
        new_data = new_df.to_numpy()
        indices = np.random.permutation(new_data.shape[0])
        i = int(split_frac * new_data.shape[0])
        training_idx, validation_idx = indices[:i], indices[i:]
        training_data, validation_data = new_data[training_idx,:], new_data[validation_idx,:]
        training_labels, validation_labels = label_list[training_idx], label_list[validation_idx]
        return training_data, training_labels, validation_data, validation_labels


if __name__ == '__main__':
    features = ['ElectronTrack_PX','ElectronTrack_PY','ElectronTrack_PZ','ElectronTrack_X','ElectronTrack_Y','ElectronTrack_Z', 
                'CaloCluster_E', 'CaloCluster_X', 'CaloCluster_Y', 'CaloCluster_Z', 'CaloCluster_Spread00',
                'CaloCluster_Spread01','CaloCluster_Spread10','CaloCluster_Spread11','CaloCluster_Covariance00',
                'CaloCluster_Covariance01','CaloCluster_Covariance02','CaloCluster_Covariance10','CaloCluster_Covariance11',
                'CaloCluster_Covariance12','CaloCluster_Covariance20','CaloCluster_Covariance21','CaloCluster_Covariance22']

    fname = 'data\\ElectronGun_1000_events.root'

    DP = DataPrep(fname,features,1000)
    df = DP.extract_data()
    mixed_data_groups = DP.generate_data_mixing(df)
    training_data, training_labels, validation_data, validation_labels = DP.prepare_data(mixed_data_groups)

    # Test Model with calo cluster info trained on 80k electron gun dataset
    with open('xgb_model_calo_info.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
    train_pred = xgb_model.predict(training_data)
    val_pred = xgb_model.predict(validation_data)
    #test_pred =  xgb_model.predict(training_data1)
    train_score = accuracy_score(training_labels,train_pred)
    val_score = accuracy_score(validation_labels,val_pred)
    test_score = 'Not explored here!' #accuracy_score(training_labels1,test_pred)
    print('train_score: ', train_score,' | ', 'Val_score: ', val_score,' | ','test_score: ', test_score)

    feat_imp = pd.Series(xgb_model.feature_importances_,features).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')