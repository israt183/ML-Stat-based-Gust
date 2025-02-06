#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 20:36:18 2022

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
Path_merged=cd+'/Combined_obsWRF/'

#Path_merged='/Volumes/Disk 2/Study/UCONN/Research/ML_WG_project/Merged_obs_WXGB/'
RW1=pd.read_csv(Path_merged+"Elev_TH_sorted_corrected_merged_obs_WRF_10.csv",sep=',')
RW2=pd.read_csv(Path_merged+"Elev_TH_sorted_corrected_merged_obs_WRF_14.csv",sep=',')
RW3=pd.read_csv(Path_merged+"Elev_TH_sorted_corrected_merged_obs_WRF_24.csv",sep=',')
RW4=pd.read_csv(Path_merged+"Elev_TH_sorted_corrected_merged_obs_WRF_13.csv",sep=',')


# Remove column 'index'
RW1=RW1.drop(['index'],axis = 1)
RW2=RW2.drop(['index'],axis = 1)
RW3=RW3.drop(['index'],axis = 1)
RW4=RW4.drop(['index'],axis = 1)

#RW_comb=pd.concat([RW1,RW2,RW3,RW4],axis=0,ignore_index=True)
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
## be caXGBule that this cell should not run multiple times. Otherwise, values of the above features will keep changing
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

#FIXME
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
Temp=np.array(row_index)
# 1st row_index(0) is the row index of the beginning of 1st event in RW_comb, 2nd row_index(1) is the row index of the beginning of 2nd event in RW_comb,
#row index(47) is the row index of the begnning of last event in RW_comb and row index(48) is the index of the end of the last event+1  

#Number of events in the dataset
Total_events=len(Temp)-1
#Number of events for test dataset
Test_storms=1
r=Total_events/Test_storms

arr=np.arange(0, Total_events, 1)



Y_test_all=pd.DataFrame()
Y_train_all=pd.DataFrame()
Y_pred_test_all=pd.DataFrame()
Y_pred_train_all=pd.DataFrame()
UPP_test_all=pd.DataFrame()
all_score=pd.DataFrame()
#j=2
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
    
    #convert “Y_train” from a Pandas “Series” object into a NumPy array for the model 
    #to accept the target training data while fitting the model
    Y_train = np.array(Y_train)
    Y_train=Y_train.ravel()
    
    stations_test=test_data['Station_x']
    stations_train=train_data['Station_x']

    Tuned_HPs={'n_estimators': 277,
               'max_depth': 6,
               'learning_rate':0.04,
               'subsample': 0.6,
               'min_child_weight': 45,
               'colsample_bytree': 0.82,
               'colsample_bylevel':0.75,
               'colsample_bynode': 0.78,
               'gamma': 12,
               'reg_lambda':28}
    
    
    from xgboost import XGBRegressor
    #FIXME
    model = XGBRegressor(random_state=10,n_jobs=-1,**Tuned_HPs)
    model.fit(X_train_copy, Y_train)
    Y_pred = model.predict(X_test_copy)
    Y_pred=Y_pred.reshape(len(Y_test),1)
    Y_test = np.asarray(Y_test)
    #need to check how the model does on train data
    Y_pred_train=model.predict(X_train_copy)
    Y_pred_train=Y_pred_train.reshape(len(Y_pred_train),1)
    
    # Error metric for test storm
    from sklearn.metrics import mean_squared_error
    MSE=mean_squared_error(Y_test,Y_pred)
    MSE=round(MSE,3)
    #print("MSE on test data:",MSE)
    # Bias= Prediction-Observation
    # This is mean bias
    BIAS = np.mean(Y_pred-Y_test)
    BIAS=round(BIAS,3)
    #print("Bias on test data:",BIAS)
    RMSE = mean_squared_error(Y_test,Y_pred)**0.5
    RMSE=round(RMSE,3)
    #print("RMSE on test data:",RMSE)
    CRMSE = (RMSE**2-BIAS**2)**0.5
    CRMSE=round(CRMSE,3)
    #print("CRMSE on test data:",CRMSE)
    MAE=mean_absolute_error(Y_test,Y_pred)
    MAE=round(MAE,3)
    #print("MAE on test data:",MAE)
    STD_obs=round(np.std(Y_test),3)
    STD_pred=round(np.std(Y_pred),3)
    Mean_obs=np.mean(Y_test)
    Mean_pred=np.mean(Y_pred)
    CV_obs=STD_obs/Mean_obs
    CV_obs=round(CV_obs,3)
    CV_pred=STD_pred/Mean_pred
    CV_pred=round(CV_pred,3)
    
    Y_test=Y_test.ravel()  
    Y_pred=Y_pred.ravel()
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_test,Y_pred)
    #print("r value on test data:", r_value)
    
    fontszt   =12
    titlesize =12
    fontsz    =16
    line_x = np.arange(-1000,10000,10)
    c_min  = 1
    c_max  = 1000
    LWIDTH=2
    trp    = 0.6
    RTT    = 25.
    def Heat_bin_plots(MINXY,MAXXY,INCR,Y_pred,Y_test,c_min,c_max,
                       xlabel_log,ylabel_log,title_log,yticks_log):
        fig, ax = plt.subplots(1)
        eps=0.01
        bins    = (np.arange(MINXY,MAXXY+INCR,step=INCR+eps),np.arange(MINXY,MAXXY+INCR,step=INCR+eps))
        img = plt.hist2d(Y_test, Y_pred,norm=mpl.colors.LogNorm(), bins=bins, cmin = 1,cmap=plt.cm.jet)
        cbar=plt.colorbar(label="Density", orientation="vertical")
        plt.clim(c_min,c_max)
        cbar.set_ticks([1,10, 100, 1000])
        cbar.set_ticklabels(["1","10", "100", "1000"])
        plt.plot(line_x, line_x,color='black',linewidth=LWIDTH)
        #slope, intercept, r_value, p_value, std_err = stats.linregress(Y_test,Y_pred)
        line_y = slope*line_x + intercept
        plt.plot(line_x, line_y,color='gray',linestyle='--',linewidth=LWIDTH)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05,0.95,"Cor. C. = "+str(round(r_value,2))+'\n' "Bias = "+str(BIAS)+'(m/s)'+'\n' "RMSE = "+str(RMSE)+'(m/s)'+
                '\n' "CRMSE = "+str(CRMSE)+'(m/s)'+'\n' "MAE="+str(MAE)+'\n' "Sd. dev. obs="+str(STD_obs)+'\n'"Sd. dev. pred="+
                str(STD_pred)+'\n' "CV of obs="+str(CV_obs)+'\n' "CV of pred gust="+str(CV_pred)+
                '(m/s)'+'\n' "N Obs. = "+str(len(Y_test)),
                va='top', transform=ax.transAxes, fontsize = 10, color='black',bbox=props)
        if title_log==1:
            #FIXME title 
            ax.set_title("Predicted wind gust(XGB_tuned) vs. observed wind gust",fontsize = titlesize)
        if xlabel_log==1:
            ax.set_xlabel("Observed wind gust(m/s)", fontsize = titlesize)
        if ylabel_log==1:
            ax.set_ylabel("Predicted wind gust(m/s)",fontsize = titlesize )
        if yticks_log==0:
            ax.set_yticklabels([])
        plt.xticks(rotation=RTT)
        ax.tick_params(axis='both',direction='in')
        plt.grid(b=None, which='major', axis='both',linestyle=':')
        ax.set_xlim([MINXY,MAXXY])
        ax.set_ylim([MINXY,MAXXY]) 
    #fig = plt.figure(figsize=(5,5))
    Heat_bin_plots(5,35,0.5,Y_pred,Y_test,c_min,c_max,1,1,1,1)
    #FIXME
    plt.savefig(cd+'/Updated_Stations_XGB_TH_as_feature/XGB_tuned_'+str(j)+'.png',dpi=300,bbox_inches='tight')
    plt.close()
    #Feature importance_permutation method
    #This method will randomly shuffle each feature and compute the change in the model’s peXGBormance. 
    #The features which impact the peXGBormance the most are the most important one.
    #The permutation based importance is computationally expensive. 
    #The permutation based method can have problem with highly-correlated features, it can report them as unimportant.
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(model, X_train_copy, Y_train,scoring='neg_mean_squared_error',random_state=10)
    sorted_idx = perm_importance.importances_mean.argsort()
    plot1 = plt.figure(1)
    plt.barh(X_train_copy.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Increase in MSE")
    plt.title("Permutation importance")
    #FIXME
    plt.savefig(cd+'/Updated_Stations_XGB_TH_as_feature/XGB_tuned_FI'+str(j)+'.png',
                dpi=300,bbox_inches='tight')
    plt.close()
    
    #to make boxplot of all permutation importance scores
    imp=perm_importance.importances_mean
    score=pd.DataFrame(X_train_copy.columns,imp)
    score=score.reset_index()
    score.columns=['Increase in MSE', 'Feature']
    #score=score[score.columns[::-1]]
    all_score=pd.concat([score,all_score],axis=0)
    
# I commented out the following because that part has already been done for RF   
# =============================================================================
#     #Evaluation of WG_UPP correponding to the test data
#     UPP_test=np.asarray(UPP_test)
#     UPP_test=UPP_test.ravel()
#     # This is mean bias of UPP predicted wind gust
#     BIAS_UPP = np.mean(UPP_test-Y_test)
#     BIAS_UPP=round(BIAS_UPP,3)
#     #print("Bias on WG_UPP:",BIAS_UPP)
#     #This is MSE of UPP predicted wind gust
#     MSE_UPP=mean_squared_error(Y_test,UPP_test)
#     MSE_UPP=round(MSE_UPP,3)
#     #print("MSE on WG_UPP:",MSE_UPP)
#     RMSE_UPP = mean_squared_error(Y_test,UPP_test)**0.5
#     RMSE_UPP=round(RMSE_UPP,3)
#     #print("RMSE on WG_UPP:",RMSE_UPP)
#     CRMSE_UPP = (RMSE_UPP**2-BIAS_UPP**2)**0.5
#     CRMSE_UPP=round(CRMSE_UPP,3)
#     #print("CRMSE on WG_UPP:",CRMSE_UPP)
#     MAE_UPP=mean_absolute_error(Y_test,UPP_test)
#     MAE_UPP=round(MAE_UPP,3)
#     #print("MAE on WG_UPP:",MAE_UPP)
#     STD_UPP=np.std(UPP_test)
#     Mean_UPP=np.mean(UPP_test)
#     CV_UPP=STD_UPP/Mean_UPP
#     CV_UPP=round(CV_UPP,3) 
#     
#     def Heat_bin_plots(MINXY,MAXXY,INCR,UPP_test,Y_test,c_min,c_max,
#                         xlabel_log,ylabel_log,title_log,yticks_log):
#         fig, ax = plt.subplots(1)
#             # I put the eps to hide the irreularity in the plot
#         eps=0.01
#         bins    = (np.arange(MINXY,MAXXY+INCR,step=INCR+eps),np.arange(MINXY,MAXXY+INCR,step=INCR+eps))
#         img = plt.hist2d(Y_test, UPP_test,norm=mpl.colors.LogNorm(), bins=bins, cmin = 1,cmap=plt.cm.jet)
#         cbar=plt.colorbar(label="Density", orientation="vertical")
#         plt.clim(c_min,c_max)
#         cbar.set_ticks([1,10, 100, 1000])
#         cbar.set_ticklabels(["1","10", "100", "1000"])
#         plt.plot(line_x, line_x,color='black',linewidth=LWIDTH)
#         slope_UPP, intercept_UPP, r_UPP, p_value, std_err = stats.linregress(Y_test,UPP_test)
#         line_y_UPP = slope_UPP*line_x + intercept_UPP
#         plt.plot(line_x, line_y_UPP,color='gray',linestyle='--',linewidth=LWIDTH)
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#         ax.text(0.05,0.95,"Cor. C. = "+str(round(r_UPP,2))+'\n' "Bias = "+str(BIAS_UPP)+'(m/s)'+
#                 '\n' "RMSE = "+str(RMSE_UPP)+'(m/s)'+'\n' "CRMSE = "+str(CRMSE_UPP)+'(m/s)'+
#                 '\n' "MAE="+str(MAE_UPP)+'(m/s)'+'\n' "Sd. dev. obs="+str(STD_obs)+'\n'"Sd. dev. UPP pred="+
#                 str(STD_UPP)+'\n' "CV of obs="+str(CV_obs)+'\n' "CV of UPP pred gust="+str(CV_UPP)+
#                 '(m/s)'+'\n' "N Obs. = "+str(len(UPP_test)),
#                 va='top', transform=ax.transAxes, fontsize = 10, color='black',bbox=props)
#         if title_log==1:
#             ax.set_title("UPP Predicted wind gust vs. observed wind gust",fontsize = titlesize)
#         if xlabel_log==1:
#             ax.set_xlabel("Observed wind gust(m/s)", fontsize = titlesize)
#         if ylabel_log==1:
#             ax.set_ylabel("UPP Predicted wind gust(m/s)",fontsize = titlesize )
#         if yticks_log==0:
#             ax.set_yticklabels([])
#         plt.xticks(rotation=RTT)
#         ax.tick_params(axis='both',direction='in')
#         plt.grid(b=None, which='major', axis='both',linestyle=':')
#         ax.set_xlim([MINXY,MAXXY])
#         ax.set_ylim([MINXY,MAXXY])
#     Heat_bin_plots(5,35,0.5,UPP_test,Y_test,c_min,c_max,1,1,1,1)
#     #FIXME
#     plt.savefig(cd+'/Stations_removed_XGB_TH_as_feature/UPP_vs_Obs_test_set_'+str(j)+'.png',
#                 dpi=300,bbox_inches='tight')
#     plt.close()
# =============================================================================
    
    Y_test=pd.DataFrame(Y_test,columns=['Obs_on_test'])
    Y_test=pd.concat([stations_test,Y_test],axis=1)
    
    Y_pred=pd.DataFrame(Y_pred,columns=['WG_pred_on_test'])
    Y_pred=pd.concat([stations_test,Y_pred],axis=1)

    Y_train=pd.DataFrame(Y_train,columns=['Obs_on_train'])
    Y_train=pd.concat([stations_train,Y_train],axis=1)

    Y_pred_train=pd.DataFrame(Y_pred_train,columns=['WG_pred_on_trainset'])
    Y_pred_train=pd.concat([stations_train,Y_pred_train],axis=1)
    
    UPP_test=pd.DataFrame(UPP_test,columns=['UPP_for_test'])
    UPP_test=pd.concat([stations_test,UPP_test],axis=1)
       
    
    #Saving the test, train, UPP and prediction dataframes as csv
    #X_train_copy.to_csv(cd+'/RF_61_diff/X_train_'+str(j)+'.csv')
    Y_train.to_csv(cd+'/Updated_Stations_XGB_TH_as_feature/Y_train_'+str(j)+'.csv')
    #X_test_copy.to_csv(cd+'/RF_61_diff/X_test_'+str(j)+'.csv')
    Y_test.to_csv(cd+'/Updated_Stations_XGB_TH_as_feature/Y_test_'+str(j)+'.csv')
    Y_pred.to_csv(cd+'/Updated_Stations_XGB_TH_as_feature/Y_pred_'+str(j)+'.csv')
    Y_pred_train.to_csv(cd+'/Updated_Stations_XGB_TH_as_feature/Y_pred_train_'+str(j)+'.csv')
    UPP_test.to_csv(cd+'/Updated_Selected_stations_XGB_TH_as_feature/UPP_test_'+str(j)+'.csv')
    
    # save the model to disk
    #filename = r'/Volumes/Disk 2/Study/UCONN/Research/ML_WG_project/ML_code_output/Tuned_model_allX.sav'
    #joblib.dump(RF,cd+'/LOSO_RF_tuned_61_storm/LOSO_'+str(j)+'.sav')
    Y_test_all=pd.concat([Y_test,Y_test_all],axis=0)
    Y_pred_test_all=pd.concat([Y_pred,Y_pred_test_all],axis=0)
    
    Y_train_all=pd.concat([Y_train,Y_train_all],axis=0)
    Y_pred_train_all=pd.concat([Y_pred_train,Y_pred_train_all],axis=0)

    
    UPP_test_all=pd.concat([UPP_test,UPP_test_all],axis=0)
    
    #Y_pred_train=pd.DataFrame(Y_pred_train,columns=['WG_pred_on_train'])
    #Y_pred_train_all=pd.concat([Y_pred_train,Y_pred_train_all],axis=0)
    print('iteration:'+str(j)+'_done')

#%%
all_score=all_score.reset_index()
all_score.to_csv(cd+'/Updated_Selected_stations_XGB_TH_as_feature/Scores_FI_XGB_tuned_61_storms.csv', index = False)

Y_test_all.to_csv(cd+'/Updated_Selected_stations_XGB_TH_as_feature/All_obs_test_XGB.csv', index = False)
Y_pred_test_all.to_csv(cd+'/Updated_Selected_stations_XGB_TH_as_feature/All_pred_test_XGB.csv', index = False)
#Y_pred_train_all.to_csv(cd+'/Stations_removed_RF_TH_as_feature/All_pred_train_RF.csv', index = False)
UPP_test_all.to_csv(cd+'/Updated_Selected_stations_XGB_TH_as_feature/All_UPP_test_XGB.csv', index = False)

Y_train_all.to_csv(cd+'/Updated_Selected_stations_XGB_TH_as_feature/All_obs_train_XGB.csv', index = False)
Y_pred_train_all.to_csv(cd+'/Updated_Selected_stations_XGB_TH_as_feature/All_pred_train_XGB.csv', index = False)



import seaborn as sns
all_score=all_score.drop(['index'],axis = 1)
Tplot = sns.boxplot(y='Increase in MSE', x='Feature',data=all_score, 
                width=0.4,palette='Set1',flierprops = dict(markerfacecolor = '0.50', markersize = 2))

#plt.legend(loc='upper left',frameon=False)


Tplot.axes.set_title("Permutation importance plot",
                    fontsize=16)
 
Tplot.set_xlabel("Features", 
                fontsize=14)
 
Tplot.set_ylabel("Increase in MSE",
                fontsize=14)
 
Tplot.tick_params(labelsize=10)
plt.xticks(rotation=90, ha='right')
# output file name
#FIXME filename
plot_file_name="FI_XGB_Tuned_61_storms.jpg"
 
# save as jpeg
Tplot.figure.savefig(cd+'/Updated_Selected_stations_XGB_TH_as_feature/'+plot_file_name,bbox_inches='tight',
                    format='jpeg',
                    dpi=300)