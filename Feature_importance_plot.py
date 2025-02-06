#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 23:14:16 2022

@author: itu
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import array as ar
from statistics import mean
from scipy import stats
import matplotlib as mpl
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle as pkl

#cd='/Volumes/Disk 2/Study/UCONN/Research/ML_WG_project/ML_code_output/FI/XGB/'
#all_score_final=pd.read_csv(cd+"Scores_FI_XGB_tuned.csv",sep=',')
coeff=pd.read_csv('all_coefficients_gauss_identity.csv',sep=',')
import seaborn as sns
all_score_final=all_score_final.drop(['index'],axis = 1)
Tplot = sns.boxplot(y='Increase in MSE', x='Feature',data=all_score_final, 
                width=0.4,palette='Set1',flierprops = dict(markerfacecolor = '0.50', markersize = 2))

plt.legend(loc='upper left',frameon=False)


Tplot.axes.set_title("Permutation importance plot",
                    fontsize=16)
 
Tplot.set_xlabel("Features", 
                fontsize=14)
 
Tplot.set_ylabel("Increase in MSE",
                fontsize=14)
 
Tplot.tick_params(labelsize=10)
plt.xticks(rotation=90, ha='right')
# output file name
#FIXME
plot_file_name="FI_XGB_tuned_new.jpg"
 
# save as jpeg
Tplot.figure.savefig(plot_file_name,bbox_inches='tight',
                    format='jpeg',
                    dpi=300)