"""
This script runs CEBRA on the NM data from NAc and makes embeddings based on reward labels.
It does this for different output embedding dimensions and makes a plot of AUC scores at each embedding dimension.
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

#-------------------------------------------------------------------------------
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


#%%
n_trials = 1765

# get the choice time 
choice_times = df_trials_ses['choice_time'][0:n_trials].to_numpy()

# %%
# dimensions to loop over  
dimensions = np.concatenate([np.arange(1,10),np.arange(10,90,10)])

# list to store auc scores at each of these embedding dimensions
mean_scores = []
errors = []

# lists to collect the embeddings
b_embeds = []
t_embeds = []


#%%
# define number of runs to get error bars
n_iterations = 5

for d in dimensions:

    # run the nm analysis and get the embeddings and labels
    t_embed, b_embed, t_labels, [rewarded,unrewarded] = cp.nm_analysis(all_nms,df_trials_ses,trace_times, choice_times, dimension=d, window_=None)

    # store the embeddings for future
    b_embeds.append(b_embed)
    t_embeds.append(t_embed)

    scores = []

    # at each dimension make a couple of runs of the log regression model to get error bars
    for i in range(n_iterations):

        # make logistic function, fit it and use it to predict the initial labels from the embedding
        logreg = LogisticRegression(random_state=42)
        logreg.fit(b_embed, t_labels)
        prediction = logreg.predict(b_embed)

        # quantify how well the embedding mirrors the labels using the auc score

        # make a precision recall curve and get the threshold
        precision, recall, threshold = precision_recall_curve(t_labels, prediction)
        threshold = np.concatenate([np.array([0]), threshold])

        # calculate the fpr and tpr for all thresholds of the classification
        fpr, tpr, threshold = roc_curve(t_labels, prediction)

        # get the auc score and append it to the list
        roc_auc = auc(fpr, tpr)
        scores.append(roc_auc)

    # store the mean and the standard deviation 
    mean_scores.append(np.mean(scores))
    errors.append(np.std(scores))
    print("COMPLETED analysis of dimension {} out of 80".format(d))

#%%
# convert list of sd values to array
errors = np.array(errors)

# print the auc score vs the embedding dimension
plt.errorbar(x=dimensions,y=mean_scores, yerr=errors, fmt='ro')
plt.xlabel("Embedding dimension")
plt.ylabel("AUC Score")
plt.title("AUC score (logistic regression) vs embedding dim")
plt.savefig('AUC vs embedding dimension.png')

#%%



