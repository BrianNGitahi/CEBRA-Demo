"""
This is a library of helper functions for the demo note-books
"""

import sys

import os # my addtion

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#import joblib as jl
import cebra.datasets
from cebra import CEBRA
import torch
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from matplotlib.collections import LineCollection
import sklearn.linear_model

#--------------------------------------------------------------------
# function to view the ideal embedding from different angles
def view_embedding(embed1, embed2, label, label_class, titles, s=0.0001, n_angles=3):

    fig1=plt.figure(figsize=(8,4*n_angles))
    gs = gridspec.GridSpec(n_angles, 2, figure=fig1)

    c = ['cool','plasma','pink','winter']

    for i, ii in enumerate(range(60,360,int(300/n_angles))):

        # create the axes
        ax1 = fig1.add_subplot(gs[1*i,0], projection='3d')
        ax1.view_init(elev=10., azim=ii) 

        ax2 = fig1.add_subplot(gs[1*i,1], projection='3d')
        ax2.view_init(elev=10., azim=ii)

        # loop over the number of labels
        for j,value in enumerate(label_class):
            
            # plot time embedding
            cebra.plot_embedding(embedding=embed1[value,:], embedding_labels=label[value], ax=ax1, markersize=s,title=titles[0],cmap=c[j])

            # plot behaviour embedding
            cebra.plot_embedding(embedding=embed2[value,:], embedding_labels=label[value], ax=ax2, markersize=s,title=titles[1],cmap=c[j])

            plt.tight_layout()

#-------------------------------------------------------------------

# function to build, train and compute an embedding
def base_embed(input, mode='time', arch = 'offset10-model-mse', dist = 'euclidean', b_label=None, temp=1, dimension=3, lr = 3e-4, d=0.1, iters = 2000):

    # build CEBRA model
    model = CEBRA(model_architecture=arch,
                         batch_size=512,
                         learning_rate=lr,
                         temperature=int(temp),
                         output_dimension = int(dimension),
                         max_iterations=int(iters),
                         distance=dist,
                         delta=int(d),
                         conditional=mode,
                         device='cuda_if_available',
                         verbose=True,
                         time_offsets=10)
    
    # train using label if it's a behaviour model
    train_size = int(input.shape[0])

    if mode == 'time':
        model.fit(input[:train_size])
    if mode == 'delta':
        model.fit(input[:train_size],b_label[:train_size])

    embedding = model.transform(input)
    return model, embedding

#--------------------------------------------------------------------