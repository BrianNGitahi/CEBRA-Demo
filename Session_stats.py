"""
This script runs the analyses on multiple sessions
"""

#%%

import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import cebra.data
import torch
import cebra.integrations
import cebra.datasets
from cebra import CEBRA
import torch
import pickle
import utils

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc


from matplotlib.collections import LineCollection
import pandas as pd


#%%

# load the dataframe
df_trials_ses = pickle.load(open('/Users/brian.gitahi/Desktop/AIND/CEBRA/Git/CEBRA-Demo/data/CO data/df.pkl', "rb"))

# load the dictionary containing the traces
traces = pickle.load(open('/Users/brian.gitahi/Desktop/AIND/CEBRA/Git/CEBRA-Demo/data/CO data/traces.pkl', "rb"))

# load the trace times
trace_times = np.load('/Users/brian.gitahi/Desktop/AIND/CEBRA/Git/CEBRA-Demo/data/CO data/Trace times.npy', allow_pickle=True)


# Combine the traces for all NMs into one 2D array
all_nms = np.array([traces[trace] for trace in traces.keys()])
all_nms = np.transpose(all_nms)

n_trials = 1765

# get the choice time 
choice_times = df_trials_ses['choice_time'][0:n_trials].to_numpy()

#%%

# Individual NMs AUC scores
ind_nm_data = utils.individual_datasets(traces_=traces)
b_embeds, t_embeds, labels, [rewarded, unrewarded] = utils.nm_analysis_2(ind_nm_data, df=df_trials_ses,trace_times=trace_times, choice_times=choice_times, title='Individual NMs')
auc_scores, sds =  utils.get_auc(b_embeds, labels)

#%%
# Missing NMs AUC scores
three_nm_data = utils.create_datasets(traces_=traces)
b3_embeds, t3_embeds, labels3, [rewarded3, unrewarded3] = utils.nm_analysis_2(three_nm_data, df=df_trials_ses,trace_times=trace_times, choice_times=choice_times, title='3 NMs')
auc3_scores, sds3 = utils.get_auc(b3_embeds, labels3)

#%%
# AUC Score for all of them
ball_embeds, tall_embeds, labels_all, [rewardeda, unrewardeda] = utils.nm_analysis_2([all_nms], df=df_trials_ses,trace_times=trace_times, choice_times=choice_times, title='ALL NMs')
auca_scores, sds_a = utils.get_auc(ball_embeds, labels_all)


#%%
# Before and after choice AUC scores + (bonus) best two embedding pairs 
b4b_embeds, b4t_embeds, labels_b4, [r, unr] = utils.nm_analysis_2(all_nms,window='before')
afb_embeds, aft_embeds, labels_af, [r_af, unr_af] = utils.nm_analysis_2(all_nms, window='after')

auc_b4_scores, sds_b4 = utils.get_auc(b4b_embeds, labels_b4)
auc_af_scores, sds_af = utils.get_auc(afb_embeds, labels_af)

#%%
# Signal Average in the 1 sec window


#%%
# bar graph of the AUC scores of all 4 individual NMs (before and after)

    # loop over the 4 NMs in a dictionary?

    # make a bar graph



fig, ax = plt.subplots(layout='constrained')

x = np.arange(0,4)
width = 0.25

# list containing AUC scores of before and after datasets of the 4 NMs

scores = {'Before': array, 'After': array2}
