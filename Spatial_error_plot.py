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
import matplotlib.colors as colors


cd=os.getcwd()


Station_info=pd.read_csv("ISD_Coords_mountWashingtonexcluded.csv",sep=',')
Station_info=Station_info.rename(columns={"File_Name" : "Station"})

df_ML=pd.read_csv('Ext_data_alt_XGB_ML_avg_RMSE_61_storms.csv',sep=',')
df_ML=df_ML[['Station','RMSE']]

df_UPP=pd.read_csv('Ext_data_alt_XGB_UPP_avg_RMSE_61_storms.csv',sep=',')
df_UPP=df_UPP[['Station','RMSE']]

Data_ML = df_ML.merge(Station_info,on='Station',how = 'inner')
Data_ML=Data_ML.reset_index(drop=True)

Data_UPP = df_UPP.merge(Station_info,on='Station',how = 'inner')
Data_UPP=Data_UPP.reset_index(drop=True)

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
#plt.show()

vmin=0
#vmax=10
# setting the vmax according to the max error of RF
vmax=6
# Create a custom colormap by combining seismic and yellow colormaps
cmap_seismic = plt.cm.seismic
cmap_new = colors.ListedColormap(['gold'])
cmap_combined = colors.ListedColormap(list(cmap_seismic(np.linspace(0, 1, 256))) + ['gold'])

norm = colors.Normalize(vmin=vmin, vmax=vmax)
#First plot
scatter1=ax.scatter(
     Data_ML["Lon"],
     Data_ML["Lat"],
     c=Data_ML["RMSE"],
     s=40,
     edgecolor='black',
     norm=norm,
# FIXME
     cmap=cmap_combined,
     #vmin=0,
     #vmax=10,
     transform=crs.PlateCarree(),
 )
#FIXME
#plt.colorbar().set_label("Avgerage MAE (m/s) of RF")
plt.colorbar(scatter1,label="Average RMSE (m/s) [XGB]",ticks=[0, 1, 2, 3, 4, 5,6],
             extend='max', extendfrac=0.1, cmap=cmap_combined)
plt.suptitle("XGB", fontsize=20,fontweight='bold', y=0.87,color='blue')
plt.savefig('New_XGB_RMSE_over_61storms.png',dpi=300,bbox_inches='tight')

# =============================================================================
# #Second plot
# figure = plt.figure(figsize=(8,6))
# ax = figure.add_subplot(1,1,1, projection=crs.Mercator())
# 
# ax.add_feature(cfeature.BORDERS)
# ax.add_feature(cfeature.COASTLINE)
# ax.set_extent([-79,-68,38 ,47], crs=crs.PlateCarree())
# ax.add_feature(cfeature.OCEAN)
# ax.add_feature(cfeature.LAND, color='white',edgecolor='black')
# ax.add_feature(cfeature.STATES,linewidth=0.5, edgecolor='black')
# ax.add_feature(cfeature.RIVERS)
# ax.gridlines()
# 
# scatter2=plt.scatter(
#     Data_UPP["Lon"],
#     Data_UPP["Lat"],
#     c=Data_UPP["RMSE"],
#     s=40,
# #FIXME    
#     cmap=cmap_combined,
#     norm=norm,
#     edgecolor='black',
#     transform=crs.PlateCarree()
# )
# 
# #plt.colorbar().set_label("Avgerage MAE (m/s) of WRF UPP ")
# plt.colorbar(scatter2,label="Avgerage RMSE (m/s)",ticks=[0, 1, 2, 3, 4, 5,6],
#              extend='max', extendfrac=0.1, cmap=cmap_combined)
# plt.suptitle("WRF-UPP", fontsize=20,fontweight='bold', y=0.87,color='blue')
# plt.savefig('New_WRF_RMSE_over_61storms.png',dpi=300,bbox_inches='tight')
# =============================================================================



