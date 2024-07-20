"""
This script was used to generate a table containing the results of the CEBRA Analyses 
of fiber photometry data using rpe, choice and reward information as labels. The output
is a table with reconstruction scores showing how well CEBRA's embeddings were able to capture
these labels. They can further be made into summary plots as show in the notebook: Session_stats_plots.
"""

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
import seaborn as sns
import re


from matplotlib.collections import LineCollection
import pandas as pd


#%% LOAD THE DATA
df_all_choice = pickle.load(open("/Users/brian.gitahi/Desktop/AIND/CEBRA/Git/CEBRA-Demo/stats_files/all_session_stats_choice.pkl", "rb"))
df_all_reward =  pickle.load(open("/Users/brian.gitahi/Desktop/AIND/CEBRA/Git/CEBRA-Demo/stats_files/all_session_stats_reward.pkl", "rb"))
df_all_rpe = pickle.load(open("/Users/brian.gitahi/Desktop/AIND/CEBRA/Git/CEBRA-Demo/stats_files/all_session_stats(rpe).pkl", "rb"))

## make sure all of them are sorted in the same order
df_all_choice_sorted = df_all_choice.sort_values(by='ses_idx').reset_index(drop=True)
df_all_reward_sorted = df_all_reward.sort_values(by='ses_idx').reset_index(drop=True)
df_all_rpe_sorted = df_all_rpe.sort_values(by='ses_idx').reset_index(drop=True)

# %%then merge them
df_choice_reward = pd.merge(df_all_choice_sorted,df_all_reward_sorted, on =['subject_ID', 'ses_idx'])
df_all_labels_stats = pd.merge(df_choice_reward, df_all_rpe_sorted, on=['ses_idx', 'subject_ID'], how='outer')

# %% remove the cumulative counts column and make a new one
df_all_labels_stats = df_all_labels_stats.drop(['ses_idx_count_x', 'ses_idx_count_y', 'ses_idx_count'], axis=1)
df_all_labels_stats['ses_idx_count'] = df_all_labels_stats.groupby('subject_ID').cumcount() + 1

#%% Save the resulting dataframe 
results_folder = cp.define_resultsDir(save_dir='stats_files')
df_all_labels_stats.to_pickle(results_folder + os.sep + 'session_stats_all_labels(211).pkl')

# %%
