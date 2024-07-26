"""script to run CEBRA analysis on shuffled neural data + labels on multiple sessions"""
#%%
import sys
import os # my addtion

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
import cebra_pack.cebra_utils as cp


from matplotlib.collections import LineCollection
import pandas as pd

#%%
##LOAD DATA
# load the dataframe that contains data from 1 session
df_trials_ses = pickle.load(open('../data/CO data/df.pkl', "rb"))
traces = pickle.load(open('../data/CO data/traces.pkl', "rb"))
trace_times = np.load('../data/CO data/Trace times.npy', allow_pickle=True)

# Combine the traces into one 2D array
all_nms = np.array([traces[trace] for trace in traces.keys()])
all_nms = np.transpose(all_nms)

# get the choice times in each trial
n_trials=1765
choice_times = df_trials_ses['choice_time'][0:n_trials].to_numpy()


#%%
## FORMAT IT
formatted_nms, reward_labels, choice_labels, n_licks, rpe_labels = cp.format_data(all_nms,df=df_trials_ses,trace_times_=trace_times, choice_times_=choice_times)
#%%
## SHUFFLE COLUMNS
shuffled_nms = formatted_nms[:, np.random.permutation(formatted_nms.shape[1])]

## BUILD TRAIN COMPUTE
shuffled_t_embed, shuffled_b_embed = cp.build_train_compute(shuffled_nms,reward_labels)


# %%
rewarded, unrewarded = cp.define_label_classes(trial_labels)