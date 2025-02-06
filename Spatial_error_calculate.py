#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:01:42 2022

@author: itu
"""
#%% 
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

#Path_merged='/Volumes/Disk 2/Study/UCONN/Research/ML_WG_project/Merged_obs_WRF/'
Path_merged=cd+'/Combined_obsWRF/'
#importing the csv files
RW1=pd.read_csv(Path_merged+"Elev_TH_sorted_corrected_merged_obs_WRF_10.csv",sep=',')
RW2=pd.read_csv(Path_merged+"Elev_TH_sorted_corrected_merged_obs_WRF_14.csv",sep=',')
RW3=pd.read_csv(Path_merged+"Elev_TH_sorted_corrected_merged_obs_WRF_24.csv",sep=',')
RW4=pd.read_csv(Path_merged+"Elev_TH_sorted_corrected_merged_obs_WRF_13.csv",sep=',')

# Remove column 'index'
RW1=RW1.drop(['index'],axis = 1)
RW2=RW2.drop(['index'],axis = 1)
RW3=RW3.drop(['index'],axis = 1)
RW4=RW4.drop(['index'],axis = 1)


RW_comb=pd.concat([RW1,RW2,RW3,RW4],axis=0,ignore_index=True)
# I will exclude the following stations from dataframe
stations=[71905099999,72040700462,72408013739,72498894704,
          72406693706,71524099999,71967099999,72520399999,
          71294099999,99728999999,99799399999
         ]

RW_comb=RW_comb[~RW_comb['Station_x'].isin(stations)]
RW_comb=RW_comb.reset_index(drop=True)

#exclude the two high gust values
Discard_WG=[50.9,39.6]
RW_comb=RW_comb[~RW_comb['WG_o'].isin(Discard_WG)]
RW_comb=RW_comb.reset_index(drop=True)
#RW_comb.reset_index(inplace=True)
#RW_comb.to_csv('RW_combined_48_events.csv', index = False)

## Converting units of the following features: PSFC(Pa),POT_2m(K),PBLH(m)
## change PSFC to kPa from Pa, PBLH to km and POT_2m to deg C
## be carfule that this cell should not run multiple times. Otherwise, values of the above features will keep changing
def to_kPa(x):
    return x/1000
RW_comb['PSFC(Pa)']= RW_comb['PSFC(Pa)'].apply(to_kPa)
def to_degC(x): 
    return x-273.15
RW_comb['POT_2m(K)']= RW_comb['POT_2m(K)'].apply(to_degC)
def to_km(x):
    return x/1000
RW_comb['PBLH(m)']= RW_comb['PBLH(m)'].apply(to_km)

## renaming the columns for which units have been converted
RW_comb = RW_comb.rename({'PSFC(Pa)': 'PSFC(kPa)', 'POT_2m(K)': 'POT_2m(C)','PBLH(m)':'PBLH(km)','ELEV(M)':'Elevation (m)'}, axis=1)
RW_comb.head()

RW_comb['Valid_Time_x'] = pd.to_datetime(RW_comb['Valid_Time_x'], format='%Y%m%d%H')
#column index of "Valid_Time_x" is 23
a=RW_comb.at[0,'Valid_Time_x']
# empty list row_index will be used to store how many rows of the dataframe belong to each storm
row_index=[]
for i in range(len(RW_comb)):
    b=RW_comb.at[i,'Valid_Time_x']
    diff=abs(a-b)
    diff_in_hours = diff.total_seconds() / 3600
    if diff_in_hours<= 48:
        continue
    else:
        a=RW_comb.at[i,'Valid_Time_x']
        row_index.append(i)

    
row_index=[0]+row_index+[len(RW_comb)]
# 1st row_index(0) is the row index of the beginning of 1st event in RW_comb, 2nd row_index(1) is the row index of the beginning of 2nd event in RW_comb,
#row index(47) is the row index of the begnning of last event in RW_comb and row index(48) is the index of the end of the last event+1  

#Number of events in the dataset
Total_events=61
#Number of events for test dataset
Test_storms=1
r=Total_events/Test_storms

arr=np.arange(0, Total_events, 1)

all_ML_error =pd.DataFrame(columns=['Station','RMSE'])
all_UPP_error=pd.DataFrame(columns=['Station','RMSE'])

All_pred=pd.DataFrame(columns=['Station','Valid_Time','WG_o','Y_pred_test'])
All_UPP=pd.DataFrame(columns=['Station','Valid_Time','WG_o','SfcWG_UPP(m/s)'])
    
for j in range(int(r)):
    #the no. of test events
    Test_events=np.arange(j*Test_storms,(j+1)*Test_storms)
    #the number of train events
    Train_events= [x for x in arr if x not in Test_events]
    # data for all the test events
    test_data=RW_comb.iloc[row_index[Test_events[0]]:row_index[j+1],:]
    test_data=test_data.reset_index(drop=True)
    RW_copy=RW_comb.copy(deep=True)
    #data for all the train events
    train_data=RW_copy.drop(RW_copy.index[list(range(row_index[Test_events[0]],row_index[j+1]))],axis=0)
    train_data=train_data.reset_index(drop=True)  
    # Creating dataframe for the input features
    X_train=pd.DataFrame(train_data.loc[:,'PSFC(kPa)':'Terrain_height(m)'])
    X_train=X_train.drop(['SfcWG_UPP(m/s)','WG_ECMWF(m/s)','Valid_Time_x','Join_Array'
                          ,'Station_y','Valid_Time_y','Obs_Time','Tdiff','WG_o','Lat','Lon','Elevation (m)'],axis = 1)
    X_test=pd.DataFrame(test_data.loc[:,'PSFC(kPa)':'Terrain_height(m)'])  
    X_test=X_test.drop(['SfcWG_UPP(m/s)','WG_ECMWF(m/s)','Valid_Time_x','Join_Array'
                          ,'Station_y','Valid_Time_y','Obs_Time','Tdiff','WG_o','Lat','Lon','Elevation (m)'],axis = 1)
    UPP_test=pd.DataFrame(test_data['SfcWG_UPP(m/s)'])
    # Creating dataframe for targets
    Y_train=pd.DataFrame(train_data['WG_o'])
    Y_test=pd.DataFrame(test_data['WG_o'])
    # Creating dataframe for the input features by dropping highly correlated attributes
    X_train= X_train.drop(['PSFC(kPa)','POT_2m(C)','T2(K)','Pot_temp_grad(1km_sfc)','Pot_temp_grad(2km_sfc)','Pot_temp_grad(PBLH_sfc)'],axis=1)
    X_test=X_test.drop(['PSFC(kPa)','POT_2m(C)','T2(K)','Pot_temp_grad(1km_sfc)','Pot_temp_grad(2km_sfc)','Pot_temp_grad(PBLH_sfc)'],axis=1)
    X_train_copy=X_train.copy(deep=True)
    #fisrt, converting the angles to radians from degree
    X_train_copy['WindDC(cos)']= np.deg2rad(X_train_copy['WindDC(degree)']
                                               .div(360).mul(360))
    X_train_copy['WindDC(cos)']=np.cos(X_train_copy['WindDC(cos)'])
    X_train_copy['WindDC(sin)']= np.deg2rad(X_train_copy['WindDC(degree)']
                                               .div(360).mul(360))
    X_train_copy['WindDC(sin)']=np.sin(X_train_copy['WindDC(sin)'])
    X_train_copy=X_train_copy.drop(['WindDC(degree)'],axis=1)
    X_test_copy=X_test.copy(deep=True)
    #fisrt, converting the angles to radians from degree
    X_test_copy['WindDC(cos)']= np.deg2rad(X_test_copy['WindDC(degree)']
                                               .div(360).mul(360))
    X_test_copy['WindDC(cos)']=np.cos(X_test_copy['WindDC(cos)'])
    X_test_copy['WindDC(sin)']= np.deg2rad(X_test_copy['WindDC(degree)']
                                               .div(360).mul(360))
    X_test_copy['WindDC(sin)']=np.sin(X_test_copy['WindDC(sin)'])
    X_test_copy=X_test_copy.drop(['WindDC(degree)'],axis=1)
    
    Y_train = np.array(Y_train)
    Y_train=Y_train.ravel()
    #Tuned hyperparameted based on 6 fold CV
# =============================================================================
#     Tuned_HPs={'n_estimators': 276,
#                'max_depth': 14,
#                'min_samples_split': 13,
#                'max_features': 'auto',
#                'bootstrap': True,
#                'max_samples': 0.18}
# =============================================================================
    Tuned_HPs={'n_estimators':277,
               'max_depth': 6,
               'learning_rate':0.04,
               'subsample':0.6,
               'min_child_weight':45,
               'eval_metric':mean_squared_error,
               'colsample_bytree':0.82,
               'colsample_bylevel':0.75,
               'colsample_bynode':0.78,
               'gamma':12,
               'reg_lambda':28,
               }
    #from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    #FIXME n_jobs
    #model = RandomForestRegressor(random_state=10,n_jobs=-1,**Tuned_HPs)
    model = XGBRegressor(random_state=10,n_jobs=-1,**Tuned_HPs)
    model.fit(X_train_copy, Y_train)
    Y_pred = model.predict(X_test_copy)
    Y_pred=pd.DataFrame(Y_pred,columns=['Y_pred_test'])
    
    Station_obs=test_data[['Station_x','Valid_Time_x','WG_o']]
    Station_obs=Station_obs.reset_index(drop=True)
    Station_obs=Station_obs.rename(columns={"Station_x" : "Station","Valid_Time_x":"Valid_Time"})
    #adding Y_pred made on test data and Station_obs from the same test event  
    Station_pred_obs=pd.concat([Station_obs,Y_pred],axis=1,join='outer')
    #adding UPP WG of test storm with the Station_obs for the same test event
    Station_UPP_obs=pd.concat([Station_obs,UPP_test],axis=1,join='outer')
    
    All_pred=pd.concat([Station_pred_obs,All_pred])
    All_pred=All_pred.reset_index(drop=True)
    
    All_UPP=pd.concat([Station_UPP_obs,All_UPP])
    All_UPP=All_UPP.reset_index(drop=True)
#%% 
#keeping the UPP and ML predictions in the same dataframe
All_pred_UPP =pd.concat([All_pred,All_UPP],axis=1,join='outer') 
All_pred_UPP = All_pred_UPP.iloc[:, [0,1,2,3,7]]

count=All_pred_UPP.groupby(['Station']).size().reset_index(name="Times")
Unique_stations=count['Station'].unique()

for x in range (len(Unique_stations)):
    Error=All_pred_UPP.loc[All_pred_UPP['Station'] == Unique_stations[x]]
    Error=Error.reset_index(drop=True)
    
    Error['Error_ML']=mean_squared_error(Error['WG_o'],Error['Y_pred_test'])**0.5
    Error['Error_UPP']=mean_squared_error(Error['WG_o'],Error['SfcWG_UPP(m/s)'])**0.5


    RMSE_ML=round(Error['Error_ML'].mean(),2)
    RMSE_UPP=round(Error['Error_UPP'].mean(),2)
    
    RMSE_ML_df=pd.DataFrame([str(Unique_stations[x]),RMSE_ML]).T
    RMSE_ML_df.columns=['Station','RMSE']
    
    RMSE_UPP_df=pd.DataFrame([str(Unique_stations[x]),RMSE_UPP]).T
    RMSE_UPP_df.columns=['Station','RMSE']
    #all_ML_error and all_UPP error wil have avg error of each station 
    all_ML_error=all_ML_error.append(RMSE_ML_df,ignore_index=True)
    all_UPP_error=all_UPP_error.append(RMSE_UPP_df,ignore_index=True)

all_ML_error.to_csv('Ext_data_alt_XGB_ML_avg_RMSE_61_storms.csv')
all_UPP_error.to_csv('Ext_data_alt_XGB_UPP_avg_RMSE_61_storms.csv')

   