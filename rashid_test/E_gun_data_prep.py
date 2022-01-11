
from sklearn import metrics
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
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
import skopt
from skopt.space import Real, Integer
from skopt import gp_minimize
from functools import partial
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
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
            #print(events_after_mask1.shape)

            #Further Remove rows with no BremClusters
            nCI = []  # nCI = no Cluster Info, i.e. remove events with no cluster info
            for i in tqdm(events_after_mask1):
                if nCaloClusters[i] == 0:
                    nCI.append(i)
            nCI = np.array(nCI)
            
            mask2_indices = np.isin(events_after_mask1,nCI)
            events_after_mask2 = events_after_mask1[~mask2_indices]
            #print(events_after_mask2.shape)

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

    def generate_data_mixing(self, df: pd.DataFrame, ratio_of_signal_to_background = 1):
        ids = df['id'].unique().tolist()
        mixed_data = []
        columns_to_replace = [
            'ElectronTrack_PX',
            'ElectronTrack_PY', 
            'ElectronTrack_PZ',
            'ElectronTrack_X',
            'ElectronTrack_Y',
            'ElectronTrack_Z']
        for id in tqdm(ids):
            running_df = df[df['id']==id]
            running_df['label'] = 1
            sampled_df = df[df['id']!=id].sample(round(len(running_df.index.tolist())*ratio_of_signal_to_background))
            for column in columns_to_replace:
                sampled_df[column] = running_df[column].head(1).to_list()[0]
            sampled_df['label'] = 0
            combined_df = pd.concat([running_df, sampled_df])
            mixed_data.append(combined_df)

        return pd.concat(mixed_data)


    def train_validate_test_split(self, df: pd.DataFrame, test_ratio=.1, validation_ratio=.2, seed=None):
        y_data = df['label']#.to_numpy()
        X_data = df.drop(['label', 'id'], axis=1)#.to_numpy()

        train_ratio = 1- (test_ratio + validation_ratio)
        x_train, x_temp, y_train, y_temp = train_test_split(X_data, y_data, stratify = y_data, test_size=1 - train_ratio, random_state=seed)

        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=test_ratio/(test_ratio + validation_ratio), shuffle = False) 

        return (x_train,y_train), (x_val,y_val), (x_test,y_test)

if __name__ == '__main__':
    features = ['ElectronTrack_PX','ElectronTrack_PY','ElectronTrack_PZ','ElectronTrack_X','ElectronTrack_Y','ElectronTrack_Z', 
                'CaloCluster_E', 'CaloCluster_X', 'CaloCluster_Y', 'CaloCluster_Z', 'CaloCluster_Spread00',
                'CaloCluster_Spread01','CaloCluster_Spread10','CaloCluster_Spread11','CaloCluster_Covariance00',
                'CaloCluster_Covariance01','CaloCluster_Covariance02','CaloCluster_Covariance10','CaloCluster_Covariance11',
                'CaloCluster_Covariance12','CaloCluster_Covariance20','CaloCluster_Covariance21','CaloCluster_Covariance22']

    fname = 'data\\ElectronGun_1000_events.root'

    DP = DataPrep(fname,features,1000)
    df = DP.extract_data()
    mixed_data_groups = DP.generate_data_mixing(df,ratio_of_signal_to_background=1)
    # training_data, training_labels, validation_data, validation_labels = DP.prepare_data(mixed_data_groups)
    (x_train,y_train), (x_val,y_val), (x_test,y_test) = DP.train_validate_test_split(mixed_data_groups,seed = 42)

    # # defining the space
    # space = [
    #     Real(0.6, 0.9, name="colsample_bylevel"),
    #     Real(0.4, 0.7, name="colsample_bytree"),
    #     Real(0.01, 1.5, name="gamma"),
    #     Real(0.0001, 1.5, name="learning_rate"),
    #     Real(0.1, 10, name="max_delta_step"),
    #     Integer(3, 15, name="max_depth"),
    #     Real(10, 500, name="min_child_weight"),
    #     Integer(10, 1500, name="n_estimators"),
    #     Real(0.1, 100, name="reg_alpha"),
    #     Real(0.1, 100, name="reg_lambda"),
    #     Real(0.4, 0.7, name="subsample"),
    # ]
     

    # # function to fit the model and return the performance of the model
    # def return_model_assessment(args, X_train, y_train, X_test):
    #     global models, train_scores, test_scores, curr_model_hyper_params
    #     params = {curr_model_hyper_params[i]: args[i] for i, j in enumerate(curr_model_hyper_params)}
    #     model = XGBClassifier(random_state=42, seed=42,eval_metric='mlogloss',use_label_encoder=False)
    #     model.set_params(**params)
    #     fitted_model = model.fit(X_train, y_train, sample_weight=None)
    #     models.append(fitted_model)
    #     train_predictions = model.predict(X_train)
    #     test_predictions = model.predict(X_test)
    #     train_score = f1_score(train_predictions, y_train)
    #     test_score = f1_score(test_predictions, y_test)
    #     train_scores.append(train_score)
    #     test_scores.append(test_score)
    #     return 1 - test_score

    # # collecting the fitted models and model performance
    # models = []
    # train_scores = []
    # test_scores = []
    # curr_model_hyper_params = ['colsample_bylevel', 'colsample_bytree', 'gamma', 'learning_rate', 'max_delta_step',
    #                         'max_depth', 'min_child_weight', 'n_estimators', 'reg_alpha', 'reg_lambda', 'subsample']
    # objective_function = partial(return_model_assessment, X_train=X_train, y_train=y_train, X_test=X_test)

    # # running the algorithm
    # n_calls = 100 # number of times you want to train your model
    # results = gp_minimize(objective_function, space, base_estimator=None, n_calls=n_calls, n_random_starts=n_calls-1, random_state=42)



    # # # Test Model with calo cluster info trained on 80k electron gun dataset
    # # with open('ML_models/xgb_model_calo_info.pkl', 'rb') as f:
    # #         xgb_model2 = pickle.load(f)
    # # 
    # xgb_model2 = models[-1]
    # train_pred = xgb_model2.predict(X_train)
    # val_pred = xgb_model2.predict(X_test)
    # #test_pred =  xgb_model.predict(training_data1)
    # train_score = accuracy_score(y_train,train_pred)
    # val_score = accuracy_score(y_test,val_pred)
    # test_score = 'Not explored here!' #accuracy_score(training_labels1,test_pred)
    # print('train_score: ', train_score,' | ', 'Val_score: ', val_score,' | ','test_score: ', test_score)

    # feat_imp = pd.Series(xgb_model2.feature_importances_,features).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    # TN, FP, FN, TP = confusion_matrix(y_test, val_pred).ravel()
    
    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP/(TP+FN)
    # # Specificity or true negative rate
    # TNR = TN/(TN+FP) 
    # # Precision or positive predictive value
    # PPV = TP/(TP+FP)
    # # Negative predictive value
    # NPV = TN/(TN+FN)
    # # Fall out or false positive rate
    # FPR = FP/(FP+TN)
    # # False negative rate
    # FNR = FN/(TP+FN)
    # # False discovery rate
    # FDR = FP/(TP+FP)

    # # Overall accuracy
    # ACC = (TP+TN)/(TP+FP+FN+TN)

    # # comparison with existing recovery algorithms
    # # 

