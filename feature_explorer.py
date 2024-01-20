import geostatspy.GSLIB as GSLIB                        # GSLIB utilies, visualization and wrapper
import geostatspy.geostats as geostats                  # GSLIB methods convert to Python      

import sys
#!{sys.executable} -m pip install --user numpy==1.21

import numpy as np                                      # ndarrys for gridded data
import pandas as pd                                     # DataFrames for tabular data
import os                        # set working directory, run executables
import matplotlib.pyplot as plt                         # for plotting
from matplotlib.colors import ListedColormap            # custom color maps
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator) # control of axes ticks
from scipy import stats                                 # summary statistics
import math                                             # trigonometry etc.
import scipy.signal as signal                           # kernel for moving window calculation
import random                                           # for randon numbers
import seaborn as sns                                   # for matrix scatter plots
from scipy import linalg                                # for linear regression
from sklearn import preprocessing                       # remove encoding error
from sklearn.feature_selection import RFE               # for recursive feature selection
from sklearn.feature_selection import mutual_info_regression # mutual information
from sklearn.linear_model import LinearRegression       # linear regression model
from sklearn.ensemble import RandomForestRegressor      # model-based feature importance
from statsmodels.stats.outliers_influence import variance_inflation_factor # variance inflation factor
plt.rc('axes', axisbelow=True)                          # girds and axes behind all plot elements
cmap = plt.cm.inferno                                   # default colormap

def partial_corr(C):                                    # partial correlation by Fabian Pedregosa-Izquierdo, f@bianp.net
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr

dataset = 0

if dataset == 0:
    #df = pd.read_csv('unconv_MV_v4.csv')               # load our data table
    df = pd.read_csv('HackathonData2024.csv') # load data from Dr. Pyrcz's GitHub respository
    #df = df.rename(columns={'Production':'Prod'})
    response = 'Prod'

    x = df.copy(deep = True); x = x.drop(response,axis='columns')
    Y = df.loc[:,response]
    
    pred = x.columns
    resp = Y.name
    
    xmin = [6.0,0.0,1.0,10.0,0.0,0.9]; xmax = [24.0,10.0,5.0,85.0,2.2,2.9]
    Ymin = 500.0; Ymax = 9000.0
    
    predlabel = ['Porosity (%)','Permeability (mD)','Acoustic Impedance (kg/m2s*10^6)','Brittleness Ratio (%)',
                 'Total Organic Carbon (%)','Vitrinite Reflectance (%)']
    resplabel = 'Normalized Initial Production (MCFPD)'
    
    predtitle = ['Porosity','Permeability','Acoustic Impedance','Brittleness Ratio',
                 'Total Organic Carbon','Vitrinite Reflectance']
    resptitle = 'Normalized Initial Production'
    
    

df.dropna(axis=0,how='any',inplace=True) 



df['TOC'] = np.where(df['TOC']<0.0, 0.0, df['TOC'])     # set TOC < 0.0 as 0.0, otherwise leave the same
df['TOC'].describe().transpose()   


'''
if dataset == 0:
    response = 'Prod'
    
    x = df.copy(deep = True); x = x.drop(['Well',response],axis='columns')
    Y = df.loc[:,response]
    
    features = x.columns.values.tolist() + [Y.name]
    pred = x.columns.values.tolist()
    resp = Y.name
    
    xmin = [6.0,0.0,1.0,10.0,0.0,0.9]; xmax = [24.0,10.0,5.0,85.0,2.2,2.9]
    Ymin = 500.0; Ymax = 9000.0
    
    predlabel = ['Porosity (%)','Permeability (mD)','Acoustic Impedance (kg/m2s*10^6)','Brittleness Ratio (%)',
                 'Total Organic Carbon (%)','Vitrinite Reflectance (%)']
    resplabel = 'Normalized Initial Production (MCFPD)'
    
    predtitle = ['Porosity','Permeability','Acoustic Impedance','Brittleness Ratio',
                 'Total Organic Carbon','Vitrinite Reflectance']
    resptitle = 'Normalized Initial Production'
    
    featurelabel = predlabel + [resplabel]
    featuretitle = predtitle + [resptitle]

dfS = pd.DataFrame()                                    # Gaussian transform of each feature, standardization to a mean of 0 and variance of 1 
dfS['Well'] = df['Well'].values
dfS['Por'],d1,d2 = geostats.nscore(df,'Por')
dfS['Perm'],d1,d2 = geostats.nscore(df,'Perm')
dfS['AI'],d1,d2 = geostats.nscore(df,'AI')
dfS['Brittle'],d1,d2 = geostats.nscore(df,'Brittle')
dfS['TOC'],d1,d2 = geostats.nscore(df,'TOC')
dfS['VR'],d1,d2 = geostats.nscore(df,'VR')
dfS['Prod'],d1,d2 = geostats.nscore(df,'Prod')
dfS.head()

dfS['c' + resp] = pd.cut(x=df[resp], bins=[0, 2000, 4000, 6000], # make a truncated response cateogorical feature
                     labels=['Low', 'Mid', 'High'])

stand_partial_correlation = partial_corr(dfS.iloc[:,1:8])

plt.subplot(154)
plt.plot(features,stand_partial_correlation,color='black')
plt.plot([0.0,0.0,0.0,0.0,0.0,0.0],'r--',linewidth = 1.0)
plt.xlabel('Predictor Features')
plt.ylabel('Partial Correlation Coefficient')
t = plt.title('Partial Correlation Coefficient')
plt.ylim(-1,1)
plt.grid(True)'''