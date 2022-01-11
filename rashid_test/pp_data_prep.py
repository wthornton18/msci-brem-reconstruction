
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



fname = 'data\\Bu2JpsiK_ee_mu1.1_1000_events.root'