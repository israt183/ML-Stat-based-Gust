#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 20:18:21 2022

@author: itu
"""
#%%
import time
start_time=time.time()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import array as ar
from statistics import mean
from scipy import stats
import matplotlib as mpl
import math
#import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials,space_eval

#%%
cd=os.getcwd()
Path_merged=cd+'/Combined_obsWRF/'
#importing the csv files
RW1=pd.read_csv(Path_merged+"Sorted_merged_obs_WRF_10.csv",sep=',')
RW2=pd.read_csv(Path_merged+"Sorted_merged_obs_WRF_14.csv",sep=',')
RW3=pd.read_csv(Path_merged+"Sorted_merged_obs_WRF_24.csv",sep=',')
RW4=pd.read_csv(Path_merged+"Sorted_merged_obs_WRF_13.csv",sep=',')


# Remove column 'index'
RW1=RW1.drop(['index'],axis = 1)
RW2=RW2.drop(['index'],axis = 1)
RW3=RW3.drop(['index'],axis = 1)
RW4=RW4.drop(['index'],axis = 1)

RW_comb=pd.concat([RW1,RW2,RW3,RW4],axis=0,ignore_index=True)
# Remove column 'index'
#RW_comb.reset_index(inplace=True)
#RW_comb.to_csv('RW_combined_48_events.csv', index = False)

#%%
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
RW_comb = RW_comb.rename({'PSFC(Pa)': 'PSFC(kPa)', 'POT_2m(K)': 'POT_2m(C)','PBLH(m)':'PBLH(km)'}, axis=1)
RW_comb.head()
#%%
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
print(row_index)

#%%    
row_index=[0]+row_index+[len(RW_comb)]
# 1st row_index(0) is the row index of the beginning of 1st event in RW_comb, 2nd row_index(1) is the row index of the beginning of 2nd event in RW_comb,
#row index(47) is the row index of the begnning of last event in RW_comb and row index(48) is the index of the end of the last event+1  
print(row_index)

#%%
#Number of events in the dataset
Total_events=60
#Number of events for test dataset
Test_storms=10
r=Total_events/Test_storms
arr=np.arange(0, Total_events, 1)
print(arr)

#%%
for j in range(int(r)):
#for j in range(2):
#FIXME j
#j=0
#the no. of test events
    Test_events=np.arange(j*Test_storms,(j+1)*Test_storms)
    print(Test_events)
    #the number of train events
    Train_events= [x for x in arr if x not in Test_events]
    print(Train_events)
    # data for all the test events
    test_data=RW_comb.iloc[row_index[Test_events[0]]:row_index[Test_events[-1]],:]
    RW_copy=RW_comb.copy(deep=True)
    #data for all the train events
    train_data=RW_copy.drop(RW_copy.index[list(range(row_index[Test_events[0]],row_index[Test_events[-1]]))],axis=0)
    # Creating dataframe for the input features
    X_train=pd.DataFrame(train_data.loc[:,'PSFC(kPa)':'Pot_temp_grad(PBLH_sfc)'])
    X_train=X_train.drop(['SfcWG_UPP(m/s)','WG_ECMWF(m/s)'],axis = 1)
    X_test=pd.DataFrame(test_data.loc[:,'PSFC(kPa)':'Pot_temp_grad(PBLH_sfc)'])
    X_test=X_test.drop(['SfcWG_UPP(m/s)','WG_ECMWF(m/s)'],axis = 1)
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
                                               .sub(1).div(360).mul(360))
    X_train_copy['WindDC(cos)']=np.cos(X_train_copy['WindDC(cos)'])
    X_train_copy['WindDC(sin)']= np.deg2rad(X_train_copy['WindDC(degree)']
                                               .sub(1).div(360).mul(360))
    X_train_copy['WindDC(sin)']=np.sin(X_train_copy['WindDC(sin)'])
    X_train_copy=X_train_copy.drop(['WindDC(degree)'],axis=1)
    X_test_copy=X_test.copy(deep=True)
    #fisrt, converting the angles to radians from degree
    X_test_copy['WindDC(cos)']= np.deg2rad(X_test_copy['WindDC(degree)']
                                               .sub(1).div(360).mul(360))
    X_test_copy['WindDC(cos)']=np.cos(X_test_copy['WindDC(cos)'])
    X_test_copy['WindDC(sin)']= np.deg2rad(X_test_copy['WindDC(degree)']
                                               .sub(1).div(360).mul(360))
    X_test_copy['WindDC(sin)']=np.sin(X_test_copy['WindDC(sin)'])
    X_test_copy=X_test_copy.drop(['WindDC(degree)'],axis=1)
    #convert “Y_train” from a Pandas “Series” object into a NumPy array for the model 
    #to accept the target training data while fitting the model
    Y_train = np.array(Y_train)
    Y_train=Y_train.ravel()
    
    #predict on default RF model
    def_XGB=XGBRegressor(random_state=10,n_jobs=-1)
    def_XGB.fit(X_train_copy,Y_train)
    Y_pred_def=def_XGB.predict(X_test_copy)
    Y_pred_def=Y_pred_def.reshape(len(Y_test),1)
    Y_test = np.asarray(Y_test)
    
    #test errors for default RF model
    MSE=mean_squared_error(Y_test,Y_pred_def)
    MSE=round(MSE,3)
    print("MSE on test data:",MSE)
    # Bias= Prediction-Observation
    # This is mean bias
    BIAS = np.mean(Y_pred_def-Y_test)
    BIAS=round(BIAS,3)
    print("Bias on test data:",BIAS)
    RMSE = mean_squared_error(Y_test,Y_pred_def)**0.5
    RMSE=round(RMSE,3)
    print("RMSE on test data:",RMSE)
    CRMSE = (RMSE**2-BIAS**2)**0.5
    CRMSE=round(CRMSE,3)
    print("CRMSE on test data:",CRMSE)
    MAE=mean_absolute_error(Y_test,Y_pred_def)
    MAE=round(MAE,3)
    print("MAE on test data:",MAE)
    
    MSE=pd.Series(MSE)
    BIAS=pd.Series(BIAS)
    RMSE=pd.Series(RMSE)
    CRMSE=pd.Series(CRMSE)
    MAE=pd.Series(MAE)
    
    Error_RF=pd.concat([MSE,BIAS,RMSE,CRMSE,MAE],axis=0,ignore_index=True)
    Error_RF=Error_RF.to_frame()
    Names=["Avg_MSE", "Avg_BIAS","Avg_RMSE","Avg_CRMSE","Avg_MAE"]
    Error_RF['Error_metric'] = Names
    #saving test error 
    Error_RF.to_csv('61_storms_avg_Error_defXGB_test_iter_'+str(j)+'.csv', index = False)
    
    #%%
    #Hyperopt 
    trials = Trials()
    #hyperparameter space
    space={'n_estimators': hp.uniformint("n_estimators", 100, 350), #default 100
           #'criterion':hp.choice("criterion", ["mse", "mae"]),  #default squared error
           'reg_lambda' :hp.uniformint("reg_lambda",1,50), # default 1
           'min_child_weight' : hp.uniformint("min_child_weight", 1, 80),# default 1
           'max_depth':hp.uniformint("max_depth", 1, 10),# default 6
           'gamma':hp.uniformint("gamma", 0, 30), # default 0
           #'min_samples_leaf':hp.uniformint("min_samples_leaf",1,5),
           'subsample':hp.uniform("subsample",0.1,1), # default 1
           'colsample_bytree':hp.uniform("colsample_bytree",0.3,1),
           'colsample_bylevel':hp.uniform("colsample_bylevel",0.3,1),
           'colsample_bynode':hp.uniform("colsample_bynode",0.3,1),
           #'max_features':hp.choice("max_features", ["sqrt","log2","auto"]),
           #'bootstrap' : hp.choice("bootstrap",[True, False]),
           #'max_samples': hp.choice("max_samples",[math.ceil(0.5*X_train_copy.shape[0]),math.ceil(0.75*X_train_copy.shape[0]),X_train_copy.shape[0]])
           'learning_rate': hp.uniform("learning_rate",0.03,0.3) # default 0.3
           }
    random_state=10
    #another way to define objective function
    # =============================================================================
    # def hyperparameter_tuning(space):
    #     model = RandomForestRegressor(random_state=random_state, **space)
    #     cv_results = -cross_val_score(model, X_train_copy, Y_train, cv=3, scoring="neg_mean_squared_error", n_jobs=-1,error_score="raise").mean()
    #     #return cv_results
    #     return {"loss": cv_results, "status": STATUS_OK, 'eval_time': time.time()}
    # best = fmin(
    #     fn=hyperparameter_tuning,
    #     space = space, 
    #     algo=tpe.suggest, 
    #     max_evals=70, 
    #     trials=trials,
    #     rstate=np.random.default_rng(10)
    # )
    # =============================================================================
    def hyperparameter_tuning(params, random_state=random_state, cv=3, X=X_train_copy, y=Y_train):
        params = {'n_estimators': int(params['n_estimators']), 
                  #'criterion':str(params['criterion']),
                  'reg_lambda': int(params['reg_lambda']), 
                  'min_child_weight': int(params['min_child_weight']),
                  'max_depth': int(params['max_depth']),
                  'gamma': int(params['gamma']),
                  #'min_weight_fraction_leaf':float(params['min_weight_fraction_leaf']),
                  #max_features':str(params['max_features']),
                  #'bootstrap':bool(params['bootstrap']),
                  'subsample':float(params['subsample']),
                  'colsample_bytree':float(params['colsample_bytree']),
                  'colsample_bylevel':float(params['colsample_bylevel']),
                  'colsample_bynode':float(params['colsample_bynode']),
                  'learning_rate':float(params['learning_rate'])
                  }
        print(params)
        model = XGBRegressor(random_state=random_state, **params,n_jobs=-1)
        cv_results = -cross_val_score(model, X_train_copy, Y_train, cv=3, scoring="neg_mean_squared_error", n_jobs=-1,error_score="raise").mean()
        #return cv_results
        return {"loss": cv_results, "status": STATUS_OK, 'eval_time': time.time()}
    best = fmin(
        fn=hyperparameter_tuning,
        space = space, 
        algo=tpe.suggest, 
        max_evals=100, 
        trials=trials,
        rstate=np.random.default_rng(10)
    )
    HPs=pd.DataFrame.from_dict(trials.vals, orient='columns', dtype=None, columns=None)
    HPs['MSE'] = trials.losses()
    HPs.to_csv('61_storms_Hyperparameters_XGB_iter_'+str(j)+'.csv')
    #get the best hyperparameters
    Opt_HP=space_eval(space, best)
    
    #%%
    # computing the score on the test set
    # =============================================================================
    # tuned_model = RandomForestRegressor(random_state=random_state, n_estimators=int(best['n_estimators']),
    #                                     max_depth=int(best['max_depth']),
    #                                     min_samples_split=int(best['min_samples_split']),max_features='sqrt',
    #                                     bootstrap=bool(best['bootstrap']),max_samples=float(best['max_samples']),n_jobs=-1)
    # =============================================================================
    tuned_model = XGBRegressor(random_state=random_state,n_jobs=-1,**Opt_HP)
                                        
    tuned_model.fit(X_train_copy,Y_train)
    Y_pred_tuned=tuned_model.predict(X_test_copy)
    Y_pred_tuned=Y_pred_tuned.reshape(len(Y_test),1)
    #test errors for tuned RF model
    MSE_tuned=mean_squared_error(Y_test,Y_pred_tuned)
    MSE_tuned=round(MSE_tuned,3)
    print("MSE on test data:",MSE_tuned)
    # Bias= Prediction-Observation
    # This is mean bias
    BIAS_tuned = np.mean(Y_pred_tuned-Y_test)
    BIAS_tuned=round(BIAS_tuned,3)
    print("Bias on test data:",BIAS_tuned)
    RMSE_tuned= mean_squared_error(Y_test,Y_pred_tuned)**0.5
    RMSE_tuned=round(RMSE_tuned,3)
    print("RMSE on test data:",RMSE_tuned)
    CRMSE_tuned = (RMSE_tuned**2-BIAS_tuned**2)**0.5
    CRMSE_tuned=round(CRMSE_tuned,3)
    print("CRMSE on test data:",CRMSE_tuned)
    MAE_tuned=mean_absolute_error(Y_test,Y_pred_tuned)
    MAE_tuned=round(MAE_tuned,3)
    print("MAE on test data:",MAE_tuned)
    
    MSE_tuned=pd.Series(MSE_tuned)
    BIAS_tuned=pd.Series(BIAS_tuned)
    RMSE_tuned=pd.Series(RMSE_tuned)
    CRMSE_tuned=pd.Series(CRMSE_tuned)
    MAE_tuned=pd.Series(MAE_tuned)
    
    Error_RF_tuned=pd.concat([MSE_tuned,BIAS_tuned,RMSE_tuned,CRMSE_tuned,MAE_tuned],axis=0,ignore_index=True)
    Error_RF_tuned=Error_RF_tuned.to_frame()
    Names=["Avg_MSE", "Avg_BIAS","Avg_RMSE","Avg_CRMSE","Avg_MAE"]
    Error_RF_tuned['Error_metric'] = Names
    #saving error on test data
    Error_RF_tuned.to_csv('61_storms_avg_Error_tunedXGB_test_j_'+str(j)+'.csv', index = False)
    
    
    #print("Best MSE {:.3f} params {}".format( hyperparameter_tuning(best), best))


