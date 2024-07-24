"""
This script runs the CEBRA analyses on multiple sessions using reward labels.
Specifically, it runs the analysis on the data before the choice time and after the choice time
and then outputs a bargraph with reconstruction performance scores for these two periods for the individual neuromodulators.
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
import cebra_pack.cebra_utils as cp 

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
df_trials_ses = pickle.load(open('../data/CO data/df.pkl', "rb"))

# load the dictionary containing the traces
traces = pickle.load(open('../data/CO data/traces.pkl', "rb"))

# load the trace times
trace_times = np.load('../data/CO data/Trace times.npy', allow_pickle=True)


# Combine the traces for all NMs into one 2D array
all_nms = np.array([traces[trace] for trace in traces.keys()])
all_nms = np.transpose(all_nms)

n_trials = 1765

# get the choice time 
choice_times = df_trials_ses['choice_time'][0:n_trials].to_numpy()

#%%

# Individual NMs AUC scores: before and after
ind_nm_data = cp.individual_datasets(traces_=traces)
b4b_embeds, t4b_embeds, labels_b4, [rewarded_b, unrewarded_b] = cp.nm_analysis_2(ind_nm_data, df=df_trials_ses,trace_times=trace_times, choice_times=choice_times, window='before', title='Individual NMs')
auc_b4_scores, sds_b4 =  cp.get_auc(b4b_embeds, labels_b4)

afb_embeds, aft_embeds, labels_af, [r_af, unr_af] = cp.nm_analysis_2(ind_nm_data, df=df_trials_ses,trace_times=trace_times, choice_times=choice_times, window='after', title='Individual NMs')
auc_af_scores, sds_af =  cp.get_auc(afb_embeds, labels_af)


#%%
# bar graph of the AUC scores of all 4 individual NMs (before and after)

# some parameters
# labels
x_labels = ( "Before" , "After" )


x = np.arange(len(x_labels))
width = 0.20
multiplier = 0


# define dictionary of the auc scores of the NMs
aucs_b4_af = {'Dopamine': (auc_b4_scores[0], auc_af_scores[0]), 
              'Norepinephrine':(auc_b4_scores[1], auc_af_scores[1]), 
              'Serotonin': (auc_b4_scores[2], auc_af_scores[2]), 
              'Acetylcholine':(auc_b4_scores[3], auc_af_scores[3])}



fig, ax = plt.subplots(layout='constrained')

# loop over the 4 NMs in a dictionary
for nm, auc_score in aucs_b4_af.items():

    # space out the two graphs
    offset = width*multiplier
    
    # make the bars
    rects = ax.bar(x+offset, np.round(auc_score,2), width, label=nm)
    ax.bar_label(rects, padding=3)

    # in case there's more than just two sets
    multiplier+=1


ax.set_ylabel('AUC Scores', fontsize=13)
ax.set_title('AUC Scores (Before/After Choice) for Different Neuromodulators', fontsize=15)
ax.set_xticks(x+width*1.5, x_labels)
ax.legend(loc='upper left', fontsize='small', ncols=2)
ax.set_ylim(0.5, 1)

plt.show()

plt.savefig('Before/After choice Individual NMs.png')
