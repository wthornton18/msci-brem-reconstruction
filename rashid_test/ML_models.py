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
from sklearn.metrics import roc_auc_score, roc_curve
import plotly.graph_objects as go
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
from sklearn.metrics import precision_recall_curve
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.options.mode.chained_assignment = None
plt.rcParams['axes.xmargin'] = 0
#pd.set_option('display.max_rows', 400)

# import sys
# import os
# # insert at 1, 0 is the script path (or '' in REPL)
# #sys.path.insert(1, 'MSci_project\\rashid_test')
# os.chdir('c:\\Users\\Rashi\\Documents\\MSci_project\\rashid_test')
import E_gun_data_prep as egdp
import Jpsi_data_prep as jdp

class ML_models:


    def __init__(self):
        pass

    
    def optimise_model(self, model):
    # defining the space
        search_space = [
            Real(0.6, 0.9, name="colsample_bylevel"),
            Real(0.4, 0.7, name="colsample_bytree"),
            Real(0.01, 1.5, name="gamma"),
            Real(0.0001, 1.5, name="learning_rate"),
            Real(0.1, 10, name="max_delta_step"),
            Integer(3, 15, name="max_depth"),
            Real(1, 500, name="min_child_weight"),
            Integer(10, 1500, name="n_estimators"),
            Real(0.1, 100, name="reg_alpha"),
            Real(0.1, 100, name="reg_lambda"),
            Real(0.4, 0.7, name="subsample"),
        ]
        

        # function to fit the model and return the performance of the model
        def return_model_assessment(args, X_train, y_train, X_test):
            global models, train_scores, test_scores, curr_model_hyper_params
            params = {curr_model_hyper_params[i]: args[i] for i, j in enumerate(curr_model_hyper_params)}
            model = XGBClassifier(random_state=42, seed=42,objective = 'binary:logistic',eval_metric='logloss',use_label_encoder=False)
            model.set_params(**params)
            fitted_model = model.fit(X_train, y_train, sample_weight=None)
            models.append(fitted_model)
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            train_score = f1_score(train_predictions, y_train)
            test_score = f1_score(test_predictions, y_test)
            train_scores.append(train_score)
            test_scores.append(test_score)
            return 1 - test_score

        # collecting the fitted models and model performance
        models = []
        train_scores = []
        test_scores = []
        curr_model_hyper_params = ['colsample_bylevel', 'colsample_bytree', 'gamma', 'learning_rate', 'max_delta_step',
                                'max_depth', 'min_child_weight', 'n_estimators', 'reg_alpha', 'reg_lambda', 'subsample']
        objective_function = partial(return_model_assessment, X_train=X_train, y_train=y_train, X_test=X_test)

        # running the algorithm
        n_calls = 100 # number of times you want to train your model
        results = gp_minimize(objective_function, space, base_estimator=None, n_calls=n_calls, n_random_starts=n_calls-1, random_state=42)


    def optimise_model2(self,X_train,y_train,X_test,y_test, ratio):
        n_repeats=10
        cv    = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=101)
        folds = [(train,test) for train, test in cv.split(X_train, y_train)]

        metrics = ['auc', 'fpr', 'tpr', 'thresholds']
        results = {
            'train': {m:[] for m in metrics},
            'val'  : {m:[] for m in metrics},
            'test' : {m:[] for m in metrics}
        }

        params = {
            'objective'   : 'binary:logistic',
            'eval_metric' : 'logloss'
        }

        dtest = xgb.DMatrix(X_test, label=y_test)
        for train, test in tqdm(folds, total=len(folds)):
            dtrain = xgb.DMatrix(X_train.iloc[train,:], label=y_train.iloc[train])
            dval   = xgb.DMatrix(X_train.iloc[test,:], label=y_train.iloc[test])
            model  = xgb.train(
                dtrain                = dtrain,
                params                = params, 
                evals                 = [(dtrain, 'train'), (dval, 'val')],
                num_boost_round       = 1000,
                verbose_eval          = False,
                early_stopping_rounds = 10,
            )
            sets = [dtrain, dval, dtest]
            for i,ds in enumerate(results.keys()):
                y_preds              = model.predict(sets[i])
                labels               = sets[i].get_label()
                fpr, tpr, thresholds = roc_curve(labels, y_preds)
                results[ds]['fpr'].append(fpr)
                results[ds]['tpr'].append(tpr)
                results[ds]['thresholds'].append(thresholds)
                results[ds]['auc'].append(roc_auc_score(labels, y_preds))
        

        kind = 'val'
        try:
            title = 'ROC Curve for val Data with signal to background ratio of: {:.2f}'.format(ratio)
        except:
            title = 'ROC Curve'
    

        c_fill      = 'rgba(52, 152, 219, 0.2)'
        c_line      = 'rgba(52, 152, 219, 0.5)'
        c_line_main = 'rgba(41, 128, 185, 1.0)'
        c_grid      = 'rgba(189, 195, 199, 0.5)'
        c_annot     = 'rgba(149, 165, 166, 0.5)'
        c_highlight = 'rgba(192, 57, 43, 1.0)'
        fpr_mean    = np.linspace(0, 1, 100)
        interp_tprs = []
        for i in range(n_repeats):
            fpr           = results[kind]['fpr'][i]
            tpr           = results[kind]['tpr'][i]
            interp_tpr    = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean     = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std      = 2*np.std(interp_tprs, axis=0)
        tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower    = tpr_mean-tpr_std
        auc          = np.mean(results[kind]['auc'])
        fig = go.Figure([
            go.Scatter(
                x          = fpr_mean,
                y          = tpr_upper,
                line       = dict(color=c_line, width=1),
                hoverinfo  = "skip",
                showlegend = False,
                name       = 'upper'),
            go.Scatter(
                x          = fpr_mean,
                y          = tpr_lower,
                fill       = 'tonexty',
                fillcolor  = c_fill,
                line       = dict(color=c_line, width=1),
                hoverinfo  = "skip",
                showlegend = False,
                name       = 'lower'),
            go.Scatter(
                x          = fpr_mean,
                y          = tpr_mean,
                line       = dict(color=c_line_main, width=2),
                hoverinfo  = "skip",
                showlegend = True,
                name       = f'AUC: {auc:.3f}')
        ])
        fig.add_shape(
            type ='line', 
            line =dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig.update_layout(
            title = title,
            template    = 'plotly_white', 
            title_x     = 0.5,
            xaxis_title = "1 - Specificity",
            yaxis_title = "Sensitivity",
            width       = 800,
            height      = 800,
            legend      = dict(
                yanchor="bottom", 
                xanchor="right", 
                x=0.95,
                y=0.01,
            )
        )
        fig.update_yaxes(
            range       = [0, 1],
            gridcolor   = c_grid,
            scaleanchor = "x", 
            scaleratio  = 1,
            linecolor   = 'black')
        fig.update_xaxes(
            range       = [0, 1],
            gridcolor   = c_grid,
            constrain   = 'domain',
            linecolor   = 'black')
        fig.show()
        return results,model

def plot_roc_curve(training_rates, validation_rates, test_rates, ratio):
    
    try:
        title = 'ROC Curve for signal to background ratio of: {:.2f}'.format(ratio)
    except:
        title = 'ROC Curve'
    
    c_fill      = 'rgba(52, 152, 219, 0.2)'
    c_line      = 'rgba(52, 152, 219, 0.5)'
    c_line_main = 'rgba(41, 128, 185, 1.0)'
    c_red       = 'rgba(255,  0,  0,  1.0)'
    c_green     = 'rgba(0,   0, 255,  1.0)'
    c_grid      = 'rgba(189, 195, 199, 0.5)'
    c_annot     = 'rgba(149, 165, 166, 0.5)'
    c_highlight = 'rgba(192, 57, 43, 1.0)'
    fig = go.Figure([
        go.Scatter(
            x          = training_rates[0],
            y          = training_rates[1],
            line       = dict(color=c_line_main, width=1),
            hoverinfo  = "skip",
            showlegend = True,
            name       = 'Training data Curve'),
        go.Scatter(
            x          = validation_rates[0],
            y          = validation_rates[1],
            line       = dict(color=c_red, width=1),
            hoverinfo  = "skip",
            showlegend = True,
            name       = 'Validation data Curve'),
        go.Scatter(
            x          = test_rates[0],
            y          = test_rates[1],
            line       = dict(color=c_green, width=1),
            hoverinfo  = "skip",
            showlegend = True,
            name       = 'Test data Curve on Jpsi data')
     

    ])
    fig.add_shape(
        type ='line', 
        line =dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(
        title = title,
        template    = 'plotly_white', 
        title_x     = 0.5,
        xaxis_title = "1 - Specificity",
        yaxis_title = "Sensitivity",
        width       = 800,
        height      = 800,
        legend      = dict(
            yanchor="bottom", 
            xanchor="right", 
            x=0.95,
            y=0.01,
        )
    )
    fig.update_yaxes(
        range       = [0, 1],
        gridcolor   = c_grid,
        scaleanchor = "x", 
        scaleratio  = 1,
        linecolor   = 'black')
    fig.update_xaxes(
        range       = [0, 1],
        gridcolor   = c_grid,
        constrain   = 'domain',
        linecolor   = 'black')
    fig.show()    

def tp_fp_rate(model, train_data, val_data, test_data ):
    try:
        train_pred_prob = model.predict(xgb.DMatrix(train_data[0]))
        val_pred_prob = model.predict(xgb.DMatrix(val_data[0]))
        test_pred_prob = model.predict(xgb.DMatrix(test_data[0]))
    except:
        train_pred_prob = model.predict_proba(train_data[0])
        val_pred_prob = model.predict_proba(val_data[0])
        test_pred_prob = model.predict_proba(test_data[0])


    lr_tp_rates_train, lr_fp_rates_train = [], []
    lr_tp_rates_val, lr_fp_rates_val = [], []
    lr_tp_rates_test, lr_fp_rates_test = [], []

    prob_thresholds= np.linspace(0,1,100)
    for p in prob_thresholds:
        y_train_preds= []
        y_val_preds = []
        y_test_preds= []

        for train_prob in train_pred_prob:
            if train_prob > p:
                y_train_preds.append(1)
            else:
                y_train_preds.append(0)
        
        for val_prob in val_pred_prob:
            if val_prob > p:
                y_val_preds.append(1)
            else:
                y_val_preds.append(0)

        for test_prob in test_pred_prob:
            if test_prob > p:
                y_test_preds.append(1)
            else:
                y_test_preds.append(0)
        
        TN1, fp_train, FN1, tp_train = confusion_matrix(train_data[1], y_train_preds).ravel()
        TN2, fp_val, FN2, tp_val = confusion_matrix(val_data[1], y_val_preds).ravel()
        TN3, fp_test, FN3, tp_test = confusion_matrix(test_data[1], y_test_preds).ravel()

        fp_train_rate = fp_train/(fp_train + TN1)
        tp_train_rate = tp_train/(tp_train + FN1)

        fp_val_rate = fp_val/(fp_val + TN2)
        tp_val_rate = tp_val/(tp_val + FN2)

        fp_test_rate = fp_test/(fp_test + TN3)
        tp_test_rate = tp_test/(tp_test + FN3)

        lr_fp_rates_train.append(fp_train_rate)
        lr_tp_rates_train.append(tp_train_rate)

        lr_fp_rates_val.append(fp_val_rate)
        lr_tp_rates_val.append(tp_val_rate)

        lr_fp_rates_test.append(fp_test_rate)
        lr_tp_rates_test.append(tp_test_rate)

    return (lr_fp_rates_train, lr_tp_rates_train) , (lr_fp_rates_val, lr_tp_rates_val) , (lr_fp_rates_test, lr_tp_rates_test)
        



if __name__ == '__main__':
    features = ['ElectronTrack_PX','ElectronTrack_PY','ElectronTrack_PZ','ElectronTrack_X','ElectronTrack_Y','ElectronTrack_Z', 
                'CaloCluster_E', 'CaloCluster_X', 'CaloCluster_Y', 'CaloCluster_Z', 'CaloCluster_Spread00',
                'CaloCluster_Spread01','CaloCluster_Spread10','CaloCluster_Spread11','CaloCluster_Covariance00',
                'CaloCluster_Covariance01','CaloCluster_Covariance02','CaloCluster_Covariance10','CaloCluster_Covariance11',
                'CaloCluster_Covariance12','CaloCluster_Covariance20','CaloCluster_Covariance21','CaloCluster_Covariance22']

    fname = 'data\\ElectronGun_1000000_events.root'

    DP = egdp.DataPrep(fname,features,2000)
    df = DP.extract_data()
    mixed_data_groups = DP.generate_data_mixing(df,1)

    (X_train,y_train), (X_val,y_val), (X_test,y_test) = DP.train_validate_test_split(mixed_data_groups, validation_ratio = 0.2,test_ratio = 0.01,seed = 42)

    ML = ML_models().optimise_model2(X_train,y_train,X_val,y_val, ratio = 1)
    
    #
    fname2 = 'data\\Bu2JpsiK_ee_1000_events.root'

    DP2 = jdp.DataPrep(fname2,features,1000)
    df2 = DP2.extract_data()
    mixed_data_groups2 = DP.generate_data_mixing(df2,1)

    (X_train2,y_train2), (X_val2,y_val2), (X_test2,y_test2) = DP.train_validate_test_split(mixed_data_groups2,validation_ratio = 0.01, test_ratio = 0.01, seed = 42)

    #ML2 = ML_models().optimise_model2(X_train2,y_train2,X_test2,y_test2)

    train_rates, val_rates, test_rates = tp_fp_rate(ML[1],(X_train,y_train), (X_val,y_val), (X_train2,y_train2))
    plot_roc_curve(train_rates, val_rates, test_rates, ratio = 1)

    print('Starting Varying Ratios of Signal to Background')
    for ratio in np.linspace(0.2,2,10):

        mixed_data_groups = DP.generate_data_mixing(df,ratio)
        (X_train,y_train), (X_val,y_val), (X_test,y_test) = DP.train_validate_test_split(mixed_data_groups, validation_ratio = 0.2,test_ratio = 0.01,seed = 42)
        ML = ML_models().optimise_model2(X_train,y_train,X_val,y_val,ratio)
        model = ML[1]
        
        train_rates, val_rates, test_rates = tp_fp_rate(model,(X_train,y_train), (X_val,y_val), (X_train2,y_train2))
        plot_roc_curve(train_rates, val_rates, test_rates, ratio)

    print('Starting increasing data size')
    for n_data in np.arange(5000,100000,10000):
        ratio = 1.0
        print('Simulating for: {}'.format(n_data))
        DP = egdp.DataPrep(fname,features,n_data)
        df = DP.extract_data()
        mixed_data_groups = DP.generate_data_mixing(df,ratio)
        (X_train,y_train), (X_val,y_val), (X_test,y_test) = DP.train_validate_test_split(mixed_data_groups, validation_ratio = 0.2,test_ratio = 0.01,seed = 42)
        ML = ML_models().optimise_model2(X_train,y_train,X_val,y_val,ratio)
        model = ML[1]
        
        train_rates, val_rates, test_rates = tp_fp_rate(model,(X_train,y_train), (X_val,y_val), (X_train2,y_train2))
        plot_roc_curve(train_rates, val_rates, test_rates, ratio)

