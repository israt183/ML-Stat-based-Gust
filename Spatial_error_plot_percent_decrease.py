#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:47:18 2022

@author: itu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import array as ar
from statistics import mean
from scipy import stats
import shapely
import cartopy.crs as crs
import cartopy.feature as cfeature
from sklearn.metrics import mean_squared_error
from matplotlib.cm import get_cmap
import matplotlib.colors


cd=os.getcwd()


Station_info=pd.read_csv("ISD_Coords_mountWashingtonexcluded.csv",sep=',')
Station_info=Station_info.rename(columns={"File_Name" : "Station"})

df_ML=pd.read_csv('Ext_data_alt_XGB_ML_avg_RMSE_61_storms.csv',sep=',')
df_ML=df_ML[['Station','RMSE']]

df_UPP=pd.read_csv('Ext_data_alt_XGB_UPP_avg_RMSE_61_storms.csv',sep=',')
df_UPP=df_UPP[['Station','RMSE']]

Data_ML = df_ML.merge(Station_info,on='Station',how = 'inner')
Data_ML=Data_ML.reset_index(drop=True)
Data_ML=Data_ML.rename(columns={'RMSE':'Error_ML'})

Data_UPP = df_UPP.merge(Station_info,on='Station',how = 'inner')
Data_UPP=Data_UPP.reset_index(drop=True)
Data_UPP=Data_UPP.rename(columns={'RMSE':'Error_UPP'})

#Data_ML['% decrease']=((Data_UPP['Error_UPP']-Data_ML['Error_ML'])/Data_UPP['Error_UPP'])*100
Data_ML['% decrease']=((Data_ML['Error_ML']-Data_UPP['Error_UPP'])/Data_UPP['Error_UPP'])*100

figure = plt.figure(figsize=(8,6))
ax = figure.add_subplot(1,1,1, projection=crs.Mercator())
# adds a stock image as a background
#ax.stock_img()
# adds national borders
ax.add_feature(cfeature.BORDERS)
# add coastlines
ax.add_feature(cfeature.COASTLINE)
#sequence for ax.set_extent: min lon,max lon,min lat,max lat
ax.set_extent([-79,-68,38 ,47,],crs=crs.PlateCarree())
#ax.set_extent([-85,-60,35 ,50,],crs=crs.PlateCarree())
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND, color='white',edgecolor='black')
ax.add_feature(cfeature.STATES,linewidth=0.5, edgecolor='black')
ax.add_feature(cfeature.RIVERS)
ax.gridlines()

vmin=-100
#vmin=0
vmax=0
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = matplotlib.cm.get_cmap('seismic', 10)
cmap_new = matplotlib.colors.ListedColormap(['gold'])
cmap_combined = matplotlib.colors.ListedColormap(list(cmap(np.linspace(0, 1, 256))) + ['gold'])

#cmap = matplotlib.cm.get_cmap('seismic', 11)
scatter1=ax.scatter(
     Data_ML["Lon"],
     Data_ML["Lat"],
     c=Data_ML["% decrease"],
     s=50,
     edgecolor='black',
     norm=norm,
     cmap=cmap_combined,
# =============================================================================
#      vmin=-100,
#      vmax=0,
# =============================================================================
     transform=crs.PlateCarree(),
 )

#ticks = np.linspace(-10, 100, 12)
ticks = np.linspace(-100, 0, 11)
plt.colorbar(scatter1,label="% change of RMSE (m/s) over WRF-UPP",ticks=ticks,
                  extend='max', extendfrac=0.1, cmap=cmap_combined)
plt.suptitle("XGB", fontsize=20,fontweight='bold', y=0.87,color='blue')
plt.savefig('New_%_change_XGB_Avg_RMSE_over_61storms.png',dpi=300,bbox_inches='tight')






