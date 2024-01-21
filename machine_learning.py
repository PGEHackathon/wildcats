"""
(1) Instantiate a model
(2) Train the model to your data
(3) Make prediction with your data
(4) Model tuning (such as cross validation)
"""
import seaborn as sns
import matplotlib.pyplot as plt
import os                                                 # to set current working directory
import math                                               # basic calculations like square root
from sklearn.model_selection import train_test_split      # train and test split
from sklearn import tree                                  # tree program from scikit learn (package for machine learning)
from sklearn.metrics import mean_squared_error            # specific measures to check our models
import pandas as pd                                       # DataFrames for tabular data
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import keras
from tensorflow.python.keras import backend as k


# git pull -> git push
def get_interest_features(x, y, z, dataframe):
    dataframe = dataframe[[x, y, z]]
    return dataframe

# Plot characteristics over 2 other variables
def create_characteristics_plot(interest_dataframe):
    sns.scatterplot(y=interest_dataframe[interest_dataframe.columns[0]],
                    x=interest_dataframe[interest_dataframe.columns[1]],
                    hue=interest_dataframe[interest_dataframe.columns[2]])
    plt.title(f"{interest_dataframe.columns[0]} over {interest_dataframe.columns[1]}");

def visualize_model(model,xfeature,x_min,x_max,yfeature,y_min,y_max,response,z_min,z_max,title,):# plots the data points and the decision tree prediction
    cmap = plt.cm.inferno
    xplot_step = (x_max - x_min)/300.0; yplot_step = (y_max - y_min)/300.0 # resolution of the model visualization
    xx, yy = np.meshgrid(np.arange(x_min, x_max, xplot_step), # set up the mesh
                     np.arange(y_min, y_max, yplot_step))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])      # predict with our trained model over the mesh
    Z = Z.reshape(xx.shape)
    plt.scatter(xfeature,yfeature,s=None, c=response, marker=None, cmap=cmap, norm=None, vmin=z_min, vmax=z_max, alpha=0.8, linewidths=0.3, edgecolors="white")
    im = plt.imshow(Z,interpolation = None,aspect = "auto",extent = [x_min,x_max,y_min,y_max], vmin = z_min, vmax = z_max,cmap = cmap)
    plt.title(title)                                       # add the labels
    plt.xlabel(xfeature.name); plt.ylabel(yfeature.name)
    plt.xlim([x_min,x_max]); plt.ylim([y_min,y_max])
    cbar = plt.colorbar(im, orientation = 'vertical')      # add the color bar
    cbar.set_label(response.name, rotation=270, labelpad=20)
    return(plt)

""" Now we have a dataframe with the features of interest """

np.random.seed(73073)
plt.subplot(121)                                          # scatter plot of the training data, 1 row x 2 columns
im = plt.scatter(X_train["Avg Close Pressure"], X_train["TVD"],s=None, c=y_train["Avg Pump Difference"], marker=None, cmap=cmap, norm=None, vmin=avg_pump_difference_min, vmax=avg_pump_difference_max, alpha=0.8, linewidths=0.3, edgecolors="black")
plt.title('Training Data: Avg Pump Difference vs. Avg Close Pressure and TVD'); plt.xlabel('Avg Close Pressure (psi)'); plt.ylabel('TVD (ft)')
cbar = plt.colorbar(im, orientation = 'vertical'); plt.xlim(avg_close_pressure_min,avg_close_pressure_max); plt.ylim(avg_TVD_min,avg_TVD_max)
cbar.set_label("Avg Pump Difference (gal/min)", rotation=270, labelpad=20)

plt.subplot(122)                                          # scatter plot of the testing data, 1 row x 2 columns
im = plt.scatter(X_test["Avg Close Pressure"],X_test["TVD"],s=None, c=y_test["Avg Pump Difference"], marker=None, cmap=cmap, norm=None, vmin=avg_pump_difference_min, vmax=avg_pump_difference_max, alpha=0.8, linewidths=0.3, edgecolors="black")
plt.title('Training Data: Avg Pump Difference vs. Avg Close Pressure and TVD'); plt.xlabel('Avg Close Pressure (psi)'); plt.ylabel('TVD (ft)')
cbar = plt.colorbar(im, orientation = 'vertical'); plt.xlim(avg_close_pressure_min,avg_close_pressure_max); plt.ylim(avg_TVD_min,avg_TVD_max)
cbar.set_label("Avg Pump Difference (gal/min)", rotation=270, labelpad=20)

plt.subplots_adjust(left=0.0, bottom=0.0, right=2.2, top=1.2, wspace=0.3, hspace=0.2)
plt.show()

# Step 1. Instantiate the Model
decision_tree_reg = tree.DecisionTreeRegressor(max_leaf_nodes = 12) # make the model object and set the hyperparameters

# Step 2: Train (Fit) the Model with Training Data
decision_tree_reg.fit(X_train.values, y_train["Avg Pump Difference"].values) # train (fit) the model with the training data
#multilinear_reg.fit(X_train["Por"].values.reshape(n_train,1), y_train["Production"]) # use this template if only 1 predictor feature

# Plot trained model over the predictor feature space - function makes a mesh, predicts over mesh and plots
plt.subplot(121)
plt = visualize_model(decision_tree_reg,X_train["Avg Close Pressure"],avg_close_pressure_min,avg_close_pressure_max,X_train["TVD"],avg_TVD_min,avg_TVD_max,
                      y_train["Avg Pump Difference"],avg_pump_difference_min,avg_pump_difference_max,'Training Data and Decision Tree Model')

plt.subplot(122)
plt = visualize_model(decision_tree_reg,X_test["Avg Close Pressure"],avg_close_pressure_min,avg_close_pressure_max,X_test["TVD"],avg_TVD_min,avg_TVD_max,
                      y_test["Avg Pump Difference"],avg_pump_difference_min,avg_pump_difference_max,'Testing Data and Decision Tree Model')

plt.subplots_adjust(left=0.0, bottom=0.0, right=2.2, top=1.2, wspace=0.3, hspace=0.2)
plt.show()
# Does this model work well?

imputed_df = pd.read_csv('ImputedValues.csv')

top5 = pd.DataFrame()
top5['Avg Pump Difference'] = imputed_df['Avg Pump Difference']
top5['PARENT_IN_ZONE_MIN_MAP_DIST'] = imputed_df['PARENT_IN_ZONE_MIN_MAP_DIST']
top5['CODEV_OUT_ZONE_MIN_HYPOT'] = imputed_df['CODEV_OUT_ZONE_MIN_HYPOT']
top5['CODEV_1050_WELL_COUNT']= imputed_df['CODEV_1050_WELL_COUNT']
top5['Pressure Gradient (psi/ft) new'] = imputed_df['Pressure Gradient (psi/ft) new']

top5.describe().transpose()
# Multilinear regression model % random forest.

transform = StandardScaler();
features = ['PARENT_IN_ZONE_MIN_MAP_DIST', 'CODEV_OUT_ZONE_MIN_HYPOT', 'CODEV_1050_WELL_COUNT', 'Pressure Gradient (psi/ft) new']

top5['Standardized_PARENT_IN_ZONE_MIN_MAP_DIST'] = transform.fit_transform(top5.loc[:,features].values)[:,0] # standardize the data featur
top5['Standardized CODEV_OUT_ZONE_MIN_HYPOT'] = transform.fit_transform(top5.loc[:,features].values)[:,1]
top5['Standardized CODEV_1050_WELL_COUNT'] = transform.fit_transform(top5.loc[:,features].values)[:,2]
top5['Standardized Pressure Gradient (psi/ft) new'] = transform.fit_transform(top5.loc[:,features].values)[:,3]
top5


X = top5[['Standardized_PARENT_IN_ZONE_MIN_MAP_DIST','Standardized CODEV_OUT_ZONE_MIN_HYPOT', 'Standardized CODEV_1050_WELL_COUNT', 'Standardized Pressure Gradient (psi/ft) new']]
y = top5['Avg Pump Difference']
