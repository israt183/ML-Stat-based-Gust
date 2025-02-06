#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 14:04:50 2022

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

cd=os.getcwd()

#importing the csv files
BIAS=pd.read_csv(cd+"/BIAS_all_trials_XGB.csv",sep=',')
RMSE=pd.read_csv(cd+"/RMSE_all_trials_XGB.csv",sep=',')
CRMSE=pd.read_csv(cd+"/CRMSE_all_trials_XGB.csv",sep=',')
MAE=pd.read_csv(cd+"/MAE_all_trials_XGB.csv",sep=',')

BIAS_avg=pd.DataFrame(columns=['Avg','Sample_size'])
RMSE_avg=pd.DataFrame(columns=['Avg','Sample_size'])
CRMSE_avg=pd.DataFrame(columns=['Avg','Sample_size'])
MAE_avg=pd.DataFrame(columns=['Avg','Sample_size'])
#BIAS_avg.columns=['Avg','Sample_size']
SS=[5,10,15,20,25,30,35,40,45]
for i in range(9):
    temp_bias = BIAS.loc[(BIAS['Sample_size']==SS[i]), 'Value'].mean()
    BIAS_avg.loc[len(BIAS_avg)]=[temp_bias, SS[i]]
    
    temp_rmse = RMSE.loc[(RMSE['Sample_size']==SS[i]), 'Value'].mean()
    RMSE_avg.loc[len(RMSE_avg)]=[temp_rmse, SS[i]]

    temp_crmse = CRMSE.loc[(CRMSE['Sample_size']==SS[i]), 'Value'].mean()
    CRMSE_avg.loc[len(CRMSE_avg)]=[temp_crmse, SS[i]]
    
    temp_mae = MAE.loc[(MAE['Sample_size']==SS[i]), 'Value'].mean()
    MAE_avg.loc[len(MAE_avg)]=[temp_mae, SS[i]] 

#%%    
import seaborn as sns

Bias_curve=sns.lineplot(x='Sample_size', y='Avg', ci=None, marker='o',
              data=BIAS_avg,lw=1)
Bias_curve.set_xlabel("Training sample size (Number of storms)", fontsize = 15)
Bias_curve.set_ylabel("Bias (m/s)", fontsize = 15)
#plt.ylim(-2, 2)
#FIXME 
#plot_file_name="LC_RF_bias.jpg"
plot_file_name="DC_XGB_bias.jpg"
Bias_curve.figure.savefig(plot_file_name,bbox_inches='tight',
                    format='jpeg',
                    dpi=300)

  
plt.show()

RMSE_curve=sns.lineplot(x='Sample_size', y='Avg', ci=None, marker='o',
              data=RMSE_avg,lw=1)
RMSE_curve.set_xlabel("Training sample size (Number of storms)", fontsize = 15)
RMSE_curve.set_ylabel("RMSE (m/s)", fontsize = 15)
#plt.ylim(0, 5) 
#FIXME
#plot_file_name="LC_RF_RMSE.jpg"
plot_file_name="DC_XGB_RMSE.jpg"
RMSE_curve.figure.savefig(plot_file_name,bbox_inches='tight',
                    format='jpeg',
                    dpi=300)
plt.show()  
    
CRMSE_curve=sns.lineplot(x='Sample_size', y='Avg', ci=None, marker='o',
              data=CRMSE_avg,lw=1)
CRMSE_curve.set_xlabel("Training sample size (Number of storms)", fontsize = 15)
CRMSE_curve.set_ylabel("CRMSE (m/s)", fontsize = 15)
#plt.ylim(0, 5) 
#FIXME
#plot_file_name="LC_RF_CRMSE.jpg"
plot_file_name="DC_XGB_CRMSE.jpg"
CRMSE_curve.figure.savefig(plot_file_name,bbox_inches='tight',
                    format='jpeg',
                    dpi=300)
plt.show()

MAE_curve=sns.lineplot(x='Sample_size', y='Avg', ci=None, marker='o',
              data=MAE_avg,lw=1)
MAE_curve.set_xlabel("Training sample size (Number of storms)", fontsize = 15)
MAE_curve.set_ylabel("MAE (m/s)", fontsize = 15)
#FIXME
#plot_file_name="LC_RF_MAE.jpg"
#plt.ylim(0, 5) 
plot_file_name="DC_XGB_MAE.jpg"
MAE_curve.figure.savefig(plot_file_name,bbox_inches='tight',
                    format='jpeg',
                    dpi=300)
    
