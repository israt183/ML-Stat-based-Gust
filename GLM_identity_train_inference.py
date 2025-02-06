#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:48:36 2022

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
Temp=np.array(row_index)
#Number of events in the dataset
Total_events=len(Temp)-1
#Number of events for test dataset
Test_storms=1
r=Total_events/Test_storms

arr=np.arange(0, Total_events, 1)

#
# to save coeffiecients
#FIXME
Y_pred_train_all=pd.DataFrame()
Y_pred_test_all=pd.DataFrame()
Y_test_all=pd.DataFrame()
all_params=pd.DataFrame()
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
    Feature_columns=X_train_copy.columns
    #Scaling the data
    #Before modeling, we need to “center” and “standardize” our data by scaling. 
    #We scale to control for the fact that different variables are measured on different scales. 
    #We scale so that each predictor can have a “fair fight” against each other in deciding importance.
    X_train_copy=np.array(X_train_copy)
    X_test_copy=np.array(X_test_copy)
    
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X_train_copy = ss.fit_transform(X_train_copy)
    X_test_copy = ss.transform(X_test_copy)
    
    X_train_copy=pd.DataFrame(X_train_copy,columns=Feature_columns)
    X_test_copy=pd.DataFrame(X_test_copy,columns=Feature_columns)
    #convert “Y_train” from a Pandas “Series” object into a NumPy array for the model 
    #to accept the target training data while fitting the model
    Y_train = np.array(Y_train)
    Y_train=Y_train.ravel()
    
    stations_test=test_data['Station_x']
    stations_train=train_data['Station_x']
    
    import statsmodels.api as sm
    exog, endog = sm.add_constant(X_train_copy), Y_train
    model = sm.GLM(endog, exog,
             family=sm.families.Gaussian(link=sm.families.links.identity()))
    glm=model.fit()
    params=glm.params
    #predicted Y values for exog=X_train on which the model was fitted
    Y_pred_train = glm.predict(exog)
    exog_test= sm.add_constant(X_test_copy)
    #predicted Y values for exog_test=X_test
    Y_pred_test = glm.predict(exog_test)
    #Y_pred=Y_pred.reshape(len(Y_test),1)
    Y_test = np.asarray(Y_test)
    Y_pred_test=Y_pred_test.ravel()
    Y_test=Y_test.ravel()
    
    # Error metrics for Y_test vs. Y predicted using X_test
    MSE=mean_squared_error(Y_test,Y_pred_test)
    MSE=round(MSE,3)
    #print("MSE on test data:",MSE)
    # Bias= Prediction-Observation
    # This is mean bias
    BIAS = np.mean(Y_pred_test-Y_test)
    BIAS=round(BIAS,3)
    #print("Bias on test data:",BIAS)
    RMSE = mean_squared_error(Y_test,Y_pred_test)**0.5
    RMSE=round(RMSE,3)
    #print("RMSE on test data:",RMSE)
    CRMSE = (RMSE**2-BIAS**2)**0.5
    CRMSE=round(CRMSE,3)
    #print("CRMSE on test data:",CRMSE)
    MAE=mean_absolute_error(Y_test,Y_pred_test)
    MAE=round(MAE,3)
    #print("MAE on test data:",MAE)
    STD_obs=round(np.std(Y_test),3)
    STD_pred=round(np.std(Y_pred_test),3)
    Mean_obs=np.mean(Y_test)
    Mean_pred=np.mean(Y_pred_test)
    CV_obs=STD_obs/Mean_obs
    CV_obs=round(CV_obs,3)
    CV_pred=STD_pred/Mean_pred
    CV_pred=round(CV_pred,3)
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_test,Y_pred_test)
    
    params_new = params.to_frame().T
    all_params=all_params.append(params_new)

    
    fontszt   =12
    titlesize =12
    fontsz    =16
    line_x = np.arange(-1000,10000,10)
    c_min  = 1
    c_max  = 1000
    LWIDTH=2
    trp    = 0.6
    RTT    = 25.
    def Heat_bin_plots(MINXY,MAXXY,INCR,Y_pred_test,Y_test,c_min,c_max,
                       xlabel_log,ylabel_log,title_log,yticks_log):
        fig, ax = plt.subplots(1)
        eps=0.01
        bins    = (np.arange(MINXY,MAXXY+INCR,step=INCR+eps),np.arange(MINXY,MAXXY+INCR,step=INCR+eps))
        img = plt.hist2d(Y_test, Y_pred_test,norm=mpl.colors.LogNorm(), bins=bins, cmin = 1,cmap=plt.cm.jet)
        cbar=plt.colorbar(label="Density", orientation="vertical")
        plt.clim(c_min,c_max)
        cbar.set_ticks([1,10, 100, 1000])
        cbar.set_ticklabels(["1","10", "100", "1000"])
        plt.plot(line_x, line_x,color='black',linewidth=LWIDTH)
        slope, intercept, r_value, p_value, std_err = stats.linregress(Y_test,Y_pred_test)
        line_y = slope*line_x + intercept
        plt.plot(line_x, line_y,color='gray',linestyle='--',linewidth=LWIDTH)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05,0.95,"Cor. C. = "+str(round(r_value,2))+'\n' "Bias = "+str(BIAS)+'(m/s)'+'\n' "MAE = "+str(MAE)+'(m/s)'+'\n' "RMSE = "+str(RMSE)+'(m/s)'+'\n' "CRMSE = "+str(CRMSE)+'(m/s)'+'\n' "N Obs. = "+str(len(Y_test)),
                va='top', transform=ax.transAxes, fontsize = 10, color='black',bbox=props)
        if title_log==1:
            ax.set_title("Predicted wind gust(GLM_Gauss) vs. observed wind gust",fontsize = titlesize)
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

    Heat_bin_plots(5,35,0.5,Y_pred_test,Y_test,c_min,c_max,1,1,1,1)
    plt.savefig(cd+'/Updated_Selected_stations_Iden_TH_as_feature/Scaled_Gauss_identity_'+str(j)+'.png',dpi=300,bbox_inches='tight')
    plt.close()
   
    #converting to dataframes so that I can save them as csv files

    #Y_train=pd.DataFrame(Y_train,columns=['WG_o'])
    #Y_pred_train=pd.DataFrame(Y_pred_train,columns=['WG_pred_on_trainset'])
    Y_test=pd.DataFrame(Y_test,columns=['WG_o_test'])
    Y_test=pd.concat([stations_test,Y_test],axis=1)
    
    Y_pred_test=pd.DataFrame(Y_pred_test,columns=['WG_pred_on_test'])
    Y_pred_test=pd.concat([stations_test,Y_pred_test],axis=1)
    
    Y_pred_train=pd.DataFrame(Y_pred_test,columns=['WG_pred_on_train'])
    Y_pred_train=pd.concat([stations_train,Y_pred_train],axis=1)
    #UPP_test=pd.DataFrame(UPP_test,columns=['UPP_for_testset'])
    #Saving the test, train, UPP and prediction dataframes as csv
    #X_train_copy.to_csv(cd+'/X_train_'+str(j)+'.csv')
    #Y_train.to_csv(cd+'/Y_train_'+str(j)+'.csv')
    #X_test_copy.to_csv(cd+'/X_test_'+str(j)+'.csv')
    Y_test.to_csv(cd+'/Updated_Selected_stations_Iden_TH_as_feature/Y_test_'+str(j)+'.csv')
    #Y_pred_train.to_csv(cd+'/Y_pred_train_'+str(j)+'.csv')
    Y_pred_test.to_csv(cd+'/Updated_Selected_stations_Iden_TH_as_feature/Y_pred_'+str(j)+'.csv')
    
    Y_pred_test_all=pd.concat([Y_pred_test,Y_pred_test_all],axis=0)
    Y_pred_train_all=pd.concat([Y_pred_train,Y_pred_train_all],axis=0)
    Y_test_all=pd.concat([Y_test,Y_test_all],axis=0)
    

    #UPP_test.to_csv(cd+'/LOSO_48storms_density_plot/UPP_test_'+str(j)+'.csv')
    
    #save the model to disk
    #filename = r'/Volumes/Disk 2/Study/UCONN/Research/ML_WG_project/ML_code_output/Tuned_model_allX.sav'
    #joblib.dump(RF,cd+'/LOSO_48storms_density_plot/LOSO_'+str(j)+'.sav')
    print('iteration:'+str(j)+'_done')
    

Y_pred_test_all.to_csv(cd+'/Updated_Selected_stations_Iden_TH_as_feature/All_pred_test_GLM_identity.csv', index = False)
Y_pred_train_all.to_csv(cd+'/Updated_Selected_stations_Iden_TH_as_feature/All_pred_train_GLM_identity.csv', index = False)
Y_test_all.to_csv(cd+'/Updated_Selected_stations_Iden_TH_as_feature/All_obs_test_GLM_identity.csv', index = False)
all_params.to_csv(cd+'/Updated_Selected_stations_Iden_TH_as_feature/Coeff_GLM_identity_scaled.csv',index = False)

coeff=all_params.drop(['const'],axis=1)
#coeff=coeff.drop('const',axis=1)
#coeff_new=coeff.T
import seaborn as sns
Tplot = sns.boxplot(data=coeff, width=0.4,palette='Set1',flierprops = dict(markerfacecolor = '0.50', markersize = 2))

#plt.legend(loc='upper left',frameon=False)

#FIXME
Tplot.axes.set_title("Feature coefficients of GLM-Identity",
                    fontsize=16)
 

Tplot.set_xlabel("Features", 
                fontsize=14)
 
Tplot.set_ylabel(" Coefficients",
                fontsize=14)
 
Tplot.tick_params(labelsize=10)
plt.xticks(rotation=90, ha='right')

#plt.xlim(0 , 43)
#plt.ylim(-0.3 , 1)
#plt.ylim(-0.03,0.1)
# output file name
#FIXME
plot_file_name="Scaled_coeff_GLM_identity_61.jpg"
 
# save as jpeg
Tplot.figure.savefig(cd+'/Updated_Selected_stations_Iden_TH_as_feature/'+plot_file_name,bbox_inches='tight',
                    format='jpeg',
                    dpi=300)