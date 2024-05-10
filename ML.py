#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:23:06 2022

@author: yidingyu
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import numpy as np

tf.random.set_seed(20220816)

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from scipy import optimize
import csv
import pickle
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


import imageio


#%%
#==============================
#===== Load data:
#==============================
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['figure.figsize'] = 12, 6
plt.rcParams.update({'font.size': 15})
plt.rcParams["lines.linewidth"] = 2
my_markersize = .30
MM_sizex = 200. #2m
MM_sizey = 200.
#-------------------
np.random.seed(201711)

d1_200 = pd.read_csv('/Users/yidingyu/npz/csv/all.csv',sep = ',')

df_data = d1_200 #pd.concat([dtg7,dtg8,dtg9,dtg10,df1,df2,df3],ignore_index = True)
print(df_data)
df_data=df_data.dropna()

#df_data = df_data[(df_data.mm1cor > 0) & (df_data.mm1cor < 10) & (df_data.mm2cor > 0) & (df_data.mm2cor < 20) & (df_data.hornI < -100) & (df_data.hornI > -220) & (df_data.beamI > 20) ]



#%%
#==============================
#===== Prepare data:
#==============================



input_list = ['mm1pixel1', 'mm1pixel2', 'mm1pixel3', 'mm1pixel4', 'mm1pixel5', 'mm1pixel6', 'mm1pixel7', 'mm1pixel8', 'mm1pixel9', 'mm1pixel10', 'mm1pixel11', 'mm1pixel12', 'mm1pixel13', 'mm1pixel14', 'mm1pixel15', 'mm1pixel16', 'mm1pixel17', 'mm1pixel18', 'mm1pixel19', 'mm1pixel20', 'mm1pixel21', 'mm1pixel22', 'mm1pixel23', 'mm1pixel24', 'mm1pixel25', 'mm1pixel26', 'mm1pixel27', 'mm1pixel28', 'mm1pixel29', 'mm1pixel30', 'mm1pixel31', 'mm1pixel32', 'mm1pixel33', 'mm1pixel34', 'mm1pixel35', 'mm1pixel36', 'mm1pixel37', 'mm1pixel38', 'mm1pixel39', 'mm1pixel40', 'mm1pixel41', 'mm1pixel42', 'mm1pixel43', 'mm1pixel44', 'mm1pixel45', 'mm1pixel46', 'mm1pixel47', 'mm1pixel48', 'mm1pixel49', 'mm1pixel50', 'mm1pixel51', 'mm1pixel52', 'mm1pixel53', 'mm1pixel54', 'mm1pixel55', 'mm1pixel56', 'mm1pixel57', 'mm1pixel58', 'mm1pixel59', 'mm1pixel60', 'mm1pixel61', 'mm1pixel62', 'mm1pixel63', 'mm1pixel64', 'mm1pixel65', 'mm1pixel66', 'mm1pixel67', 'mm1pixel68', 'mm1pixel69', 'mm1pixel70', 'mm1pixel71', 'mm1pixel72', 'mm1pixel73', 'mm1pixel74', 'mm1pixel75', 'mm1pixel76', 'mm1pixel77', 'mm1pixel78', 'mm1pixel79', 'mm1pixel80', 'mm1pixel81',
              'mm2pixel1', 'mm2pixel2', 'mm2pixel3', 'mm2pixel4', 'mm2pixel5', 'mm2pixel6', 'mm2pixel7', 'mm2pixel8', 'mm2pixel9', 'mm2pixel10', 'mm2pixel11', 'mm2pixel12', 'mm2pixel13', 'mm2pixel14', 'mm2pixel15', 'mm2pixel16', 'mm2pixel17', 'mm2pixel18', 'mm2pixel19', 'mm2pixel20', 'mm2pixel21', 'mm2pixel22', 'mm2pixel23', 'mm2pixel24', 'mm2pixel25', 'mm2pixel26', 'mm2pixel27', 'mm2pixel28', 'mm2pixel29', 'mm2pixel30', 'mm2pixel31', 'mm2pixel32', 'mm2pixel33', 'mm2pixel34', 'mm2pixel35', 'mm2pixel36', 'mm2pixel37', 'mm2pixel38', 'mm2pixel39', 'mm2pixel40', 'mm2pixel41', 'mm2pixel42', 'mm2pixel43', 'mm2pixel44', 'mm2pixel45', 'mm2pixel46', 'mm2pixel47', 'mm2pixel48', 'mm2pixel49', 'mm2pixel50', 'mm2pixel51', 'mm2pixel52', 'mm2pixel53', 'mm2pixel54', 'mm2pixel55', 'mm2pixel56', 'mm2pixel57', 'mm2pixel58', 'mm2pixel59', 'mm2pixel60', 'mm2pixel61', 'mm2pixel62', 'mm2pixel63', 'mm2pixel64', 'mm2pixel65', 'mm2pixel66', 'mm2pixel67', 'mm2pixel68', 'mm2pixel69', 'mm2pixel70', 'mm2pixel71', 'mm2pixel72', 'mm2pixel73', 'mm2pixel74', 'mm2pixel75', 'mm2pixel76', 'mm2pixel77', 'mm2pixel78', 'mm2pixel79', 'mm2pixel80', 'mm2pixel81', 
              'mm3pixel1', 'mm3pixel2', 'mm3pixel3', 'mm3pixel4', 'mm3pixel5', 'mm3pixel6', 'mm3pixel7', 'mm3pixel8', 'mm3pixel9', 'mm3pixel10', 'mm3pixel11', 'mm3pixel12', 'mm3pixel13', 'mm3pixel14', 'mm3pixel15', 'mm3pixel16', 'mm3pixel17', 'mm3pixel18', 'mm3pixel19', 'mm3pixel20', 'mm3pixel21', 'mm3pixel22', 'mm3pixel23', 'mm3pixel24', 'mm3pixel25', 'mm3pixel26', 'mm3pixel27', 'mm3pixel28', 'mm3pixel29', 'mm3pixel30', 'mm3pixel31', 'mm3pixel32', 'mm3pixel33', 'mm3pixel34', 'mm3pixel35', 'mm3pixel36', 'mm3pixel37', 'mm3pixel38', 'mm3pixel39', 'mm3pixel40', 'mm3pixel41', 'mm3pixel42', 'mm3pixel43', 'mm3pixel44', 'mm3pixel45', 'mm3pixel46', 'mm3pixel47', 'mm3pixel48', 'mm3pixel49', 'mm3pixel50', 'mm3pixel51', 'mm3pixel52', 'mm3pixel53', 'mm3pixel54', 'mm3pixel55', 'mm3pixel56', 'mm3pixel57', 'mm3pixel58', 'mm3pixel59', 'mm3pixel60', 'mm3pixel61', 'mm3pixel62', 'mm3pixel63', 'mm3pixel64', 'mm3pixel65', 'mm3pixel66', 'mm3pixel67', 'mm3pixel68', 'mm3pixel69', 'mm3pixel70', 'mm3pixel71', 'mm3pixel72', 'mm3pixel73', 'mm3pixel74', 'mm3pixel75', 'mm3pixel76', 'mm3pixel77', 'mm3pixel78', 'mm3pixel79', 'mm3pixel80', 'mm3pixel81']

# input_list = ['mm1pixel1', 'mm1pixel2', 'mm1pixel3', 'mm1pixel4', 'mm1pixel5', 'mm1pixel6', 'mm1pixel7', 'mm1pixel8', 'mm1pixel9', 'mm1pixel10', 'mm1pixel11', 'mm1pixel12', 'mm1pixel13', 'mm1pixel14', 'mm1pixel15', 'mm1pixel16', 'mm1pixel17', 'mm1pixel18', 'mm1pixel19', 'mm1pixel20', 'mm1pixel21', 'mm1pixel22', 'mm1pixel23', 'mm1pixel24', 'mm1pixel25', 'mm1pixel26', 'mm1pixel27', 'mm1pixel28', 'mm1pixel29', 'mm1pixel30', 'mm1pixel31', 'mm1pixel32', 'mm1pixel33', 'mm1pixel34', 'mm1pixel35', 'mm1pixel36', 'mm1pixel37', 'mm1pixel38', 'mm1pixel39', 'mm1pixel40', 'mm1pixel41', 'mm1pixel42', 'mm1pixel43', 'mm1pixel44', 'mm1pixel45', 'mm1pixel46', 'mm1pixel47', 'mm1pixel48', 'mm1pixel49', 'mm1pixel50', 'mm1pixel51', 'mm1pixel52', 'mm1pixel53', 'mm1pixel54', 'mm1pixel55', 'mm1pixel56', 'mm1pixel57', 'mm1pixel58', 'mm1pixel59', 'mm1pixel60', 'mm1pixel61', 'mm1pixel62', 'mm1pixel63', 'mm1pixel64', 'mm1pixel65', 'mm1pixel66', 'mm1pixel67', 'mm1pixel68', 'mm1pixel69', 'mm1pixel70', 'mm1pixel71', 'mm1pixel72', 'mm1pixel73', 'mm1pixel74', 'mm1pixel75', 'mm1pixel76', 'mm1pixel77', 'mm1pixel78', 'mm1pixel79', 'mm1pixel80', 'mm1pixel81']
       




# input_list = ['mm3pixel1', 'mm3pixel2', 'mm3pixel3', 'mm3pixel4', 'mm3pixel5', 'mm3pixel6', 'mm3pixel7', 'mm3pixel8', 'mm3pixel9', 'mm3pixel10', 'mm3pixel11', 'mm3pixel12', 'mm3pixel13', 'mm3pixel14', 'mm3pixel15', 'mm3pixel16', 'mm3pixel17', 'mm3pixel18', 'mm3pixel19', 'mm3pixel20', 'mm3pixel21', 'mm3pixel22', 'mm3pixel23', 'mm3pixel24', 'mm3pixel25', 'mm3pixel26', 'mm3pixel27', 'mm3pixel28', 'mm3pixel29', 'mm3pixel30', 'mm3pixel31', 'mm3pixel32', 'mm3pixel33', 'mm3pixel34', 'mm3pixel35', 'mm3pixel36', 'mm3pixel37', 'mm3pixel38', 'mm3pixel39', 'mm3pixel40', 'mm3pixel41', 'mm3pixel42', 'mm3pixel43', 'mm3pixel44', 'mm3pixel45', 'mm3pixel46', 'mm3pixel47', 'mm3pixel48', 'mm3pixel49', 'mm3pixel50', 'mm3pixel51', 'mm3pixel52', 'mm3pixel53', 'mm3pixel54', 'mm3pixel55', 'mm3pixel56', 'mm3pixel57', 'mm3pixel58', 'mm3pixel59', 'mm3pixel60', 'mm3pixel61', 'mm3pixel62', 'mm3pixel63', 'mm3pixel64', 'mm3pixel65', 'mm3pixel66', 'mm3pixel67', 'mm3pixel68', 'mm3pixel69', 'mm3pixel70', 'mm3pixel71', 'mm3pixel72', 'mm3pixel73', 'mm3pixel74', 'mm3pixel75', 'mm3pixel76', 'mm3pixel77', 'mm3pixel78', 'mm3pixel79', 'mm3pixel80', 'mm3pixel81']
              
lern_var = ['nuray_1', 'nuray_2', 'nuray_3', 'nuray_4', 'nuray_5', 'nuray_6', 'nuray_7', 'nuray_8', 'nuray_9', 'nuray_10', 'nuray_11', 'nuray_12', 'nuray_13', 'nuray_14', 'nuray_15', 'nuray_16', 'nuray_17', 'nuray_18', 'nuray_19', 'nuray_20', 'nuray_21', 'nuray_22', 'nuray_23', 'nuray_24', 'nuray_25', 'nuray_26', 'nuray_27', 'nuray_28', 'nuray_29', 'nuray_30', 'nuray_31', 'nuray_32', 'nuray_33', 'nuray_34', 'nuray_35', 'nuray_36', 'nuray_37', 'nuray_38', 'nuray_39', 'nuray_40', 'nuray_41', 'nuray_42', 'nuray_43', 'nuray_44', 'nuray_45', 'nuray_46', 'nuray_47', 'nuray_48', 'nuray_49', 'nuray_50'] 


# lern_var = [ 'nuray_41', 'nuray_42', 'nuray_43', 'nuray_44', 'nuray_45', 'nuray_46', 'nuray_47', 'nuray_48', 'nuray_49', 'nuray_50'] 

#%%



# input_list = ['mm1pixel1', 'mm1pixel2', 'mm1pixel3', 'mm1pixel4', 'mm1pixel5', 'mm1pixel6', 'mm1pixel7', 'mm1pixel8', 'mm1pixel9', 'mm1pixel10', 'mm1pixel11', 'mm1pixel12', 'mm1pixel13', 'mm1pixel14', 'mm1pixel15', 'mm1pixel16', 'mm1pixel17', 'mm1pixel18', 'mm1pixel19', 'mm1pixel20', 'mm1pixel21', 'mm1pixel22', 'mm1pixel23', 'mm1pixel24', 'mm1pixel25', 'mm1pixel26', 'mm1pixel27', 'mm1pixel28', 'mm1pixel29', 'mm1pixel30', 'mm1pixel31', 'mm1pixel32', 'mm1pixel33', 'mm1pixel34', 'mm1pixel35', 'mm1pixel36', 'mm1pixel37', 'mm1pixel38', 'mm1pixel39', 'mm1pixel40', 'mm1pixel41', 'mm1pixel42', 'mm1pixel43', 'mm1pixel44', 'mm1pixel45', 'mm1pixel46', 'mm1pixel47', 'mm1pixel48', 'mm1pixel49', 'mm1pixel50', 'mm1pixel51', 'mm1pixel52', 'mm1pixel53', 'mm1pixel54', 'mm1pixel55', 'mm1pixel56', 'mm1pixel57', 'mm1pixel58', 'mm1pixel59', 'mm1pixel60', 'mm1pixel61', 'mm1pixel62', 'mm1pixel63', 'mm1pixel64', 'mm1pixel65', 'mm1pixel66', 'mm1pixel67', 'mm1pixel68', 'mm1pixel69', 'mm1pixel70', 'mm1pixel71', 'mm1pixel72', 'mm1pixel73', 'mm1pixel74', 'mm1pixel75', 'mm1pixel76', 'mm1pixel77', 'mm1pixel78', 'mm1pixel79', 'mm1pixel80', 'mm1pixel81']
              


# lern_var = ['nuray_1','nuray_5','nuray_10','nuray_20','nuray_40'] 

# lern_var = 'hornI(kA)'
data_x = df_data[input_list]/20000
 #
 
 # data_x = df_data[input_list]/df_data['mm1pixel45']
 
 
 
 
 #%%
 # pixelsMM1 = list(df_data.columns)[5:86]
 # pixelsMM2 = list(df_data.columns)[86:167]
 # pixelsMM3 = list(df_data.columns)[167:248]


#%%
 # dfMM1 = df_data[pixelsMM1]/20000
 # dfMM2 = df_data[pixelsMM2]/4000
 # dfMM3 = df_data[pixelsMM3]/200

 

#%%


# data_x = dfMM1+dfMM2+dfMM3








 #%%

# Y = [df_data['hornI(kA)'].values,df_data['hptgt(cm)'].values,df_data['vptgt(cm)'].values,df_data['spot_size(cm)'].values]

X_train, X_test, y_train, y_test = train_test_split(data_x,df_data[lern_var],test_size=0.3, shuffle = True)

#print(X_train)
print(y_train)

train_err, test_err = [],[]

print(X_train.shape); print(y_train.shape)

#%%

#===================
#--- Define a model:
#===================
model = Sequential()
# ki = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

# model.add(Dense(300, input_dim=81, kernel_initializer=ki,activation='tanh'))
model.add(Dense(300, input_dim=81*3, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.summary()
opt = tf.keras.optimizers.Adam(lr=0.001)
model.compile(loss='mse', optimizer=opt)





#%%
history = model.fit(X_train,y_train,validation_data = (X_test,y_test), epochs=3000, batch_size=100)






#%%





plt.figure(figsize = (9,7))
plt.title('mse loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.plot(list(history.history.values())[0],'b-.')
plt.plot(list(history.history.values())[1],'r-.')
plt.legend(["Training", "Validation"], fontsize = 15)
plt.yscale('log')






#%%

# loss1= list(history.history.values())[0]
# #%%
# loss2= list(history.history.values())[0]


# #%%
# loss3= list(history.history.values())[0]

#%%

# plt.figure(figsize = (9,7))
# plt.title('mse loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.plot(loss2,'b-.')
# plt.plot(loss1,'r-.')
# plt.plot(loss3,'y-.')
# plt.legend(["row data", "row data/20000", "row data/80000"], fontsize = 15)
# plt.yscale('log')





#%%
#================
#---  Save model:
#================
model.save('/Users/yidingyu/npz/csv/nuray_test_new')











#%%

 x = [n for n in range(1,len(X_test)+1)]
 predictions = model.predict(X_test)
 testy = np.array(y_test)
 plt.figure(figsize = (9,7))
 plt.plot(x,predictions[:,5] , linewidth = 3)
 plt.plot(x, testy[:,5], linewidth = 3)
 plt.xlabel('index')
 plt.ylabel('nuray_46')
 plt.legend(["Prediction", "MC-Data"], fontsize = 15)
 # plt.ylim(-2.1,2.1)
 # plt.legend(["hptgt", "vptgt"], fontsize = 15)
 plt.grid()






#%%

d2_200 = pd.read_csv('/Users/yidingyu/npz/csv/nuray_test_50.csv',sep = ',')

df_data_2 = d2_200 #pd.concat([dtg7,dtg8,dtg9,dtg10,df1,df2,df3],ignore_index = True)
print(df_data_2)
df_data_2=df_data_2.dropna()


# input_list = ['mm1pixel1', 'mm1pixel2', 'mm1pixel3', 'mm1pixel4', 'mm1pixel5', 'mm1pixel6', 'mm1pixel7', 'mm1pixel8', 'mm1pixel9', 'mm1pixel10', 'mm1pixel11', 'mm1pixel12', 'mm1pixel13', 'mm1pixel14', 'mm1pixel15', 'mm1pixel16', 'mm1pixel17', 'mm1pixel18', 'mm1pixel19', 'mm1pixel20', 'mm1pixel21', 'mm1pixel22', 'mm1pixel23', 'mm1pixel24', 'mm1pixel25', 'mm1pixel26', 'mm1pixel27', 'mm1pixel28', 'mm1pixel29', 'mm1pixel30', 'mm1pixel31', 'mm1pixel32', 'mm1pixel33', 'mm1pixel34', 'mm1pixel35', 'mm1pixel36', 'mm1pixel37', 'mm1pixel38', 'mm1pixel39', 'mm1pixel40', 'mm1pixel41', 'mm1pixel42', 'mm1pixel43', 'mm1pixel44', 'mm1pixel45', 'mm1pixel46', 'mm1pixel47', 'mm1pixel48', 'mm1pixel49', 'mm1pixel50', 'mm1pixel51', 'mm1pixel52', 'mm1pixel53', 'mm1pixel54', 'mm1pixel55', 'mm1pixel56', 'mm1pixel57', 'mm1pixel58', 'mm1pixel59', 'mm1pixel60', 'mm1pixel61', 'mm1pixel62', 'mm1pixel63', 'mm1pixel64', 'mm1pixel65', 'mm1pixel66', 'mm1pixel67', 'mm1pixel68', 'mm1pixel69', 'mm1pixel70', 'mm1pixel71', 'mm1pixel72', 'mm1pixel73', 'mm1pixel74', 'mm1pixel75', 'mm1pixel76', 'mm1pixel77', 'mm1pixel78', 'mm1pixel79', 'mm1pixel80', 'mm1pixel81']
  

input_list = ['mm1pixel1', 'mm1pixel2', 'mm1pixel3', 'mm1pixel4', 'mm1pixel5', 'mm1pixel6', 'mm1pixel7', 'mm1pixel8', 'mm1pixel9', 'mm1pixel10', 'mm1pixel11', 'mm1pixel12', 'mm1pixel13', 'mm1pixel14', 'mm1pixel15', 'mm1pixel16', 'mm1pixel17', 'mm1pixel18', 'mm1pixel19', 'mm1pixel20', 'mm1pixel21', 'mm1pixel22', 'mm1pixel23', 'mm1pixel24', 'mm1pixel25', 'mm1pixel26', 'mm1pixel27', 'mm1pixel28', 'mm1pixel29', 'mm1pixel30', 'mm1pixel31', 'mm1pixel32', 'mm1pixel33', 'mm1pixel34', 'mm1pixel35', 'mm1pixel36', 'mm1pixel37', 'mm1pixel38', 'mm1pixel39', 'mm1pixel40', 'mm1pixel41', 'mm1pixel42', 'mm1pixel43', 'mm1pixel44', 'mm1pixel45', 'mm1pixel46', 'mm1pixel47', 'mm1pixel48', 'mm1pixel49', 'mm1pixel50', 'mm1pixel51', 'mm1pixel52', 'mm1pixel53', 'mm1pixel54', 'mm1pixel55', 'mm1pixel56', 'mm1pixel57', 'mm1pixel58', 'mm1pixel59', 'mm1pixel60', 'mm1pixel61', 'mm1pixel62', 'mm1pixel63', 'mm1pixel64', 'mm1pixel65', 'mm1pixel66', 'mm1pixel67', 'mm1pixel68', 'mm1pixel69', 'mm1pixel70', 'mm1pixel71', 'mm1pixel72', 'mm1pixel73', 'mm1pixel74', 'mm1pixel75', 'mm1pixel76', 'mm1pixel77', 'mm1pixel78', 'mm1pixel79', 'mm1pixel80', 'mm1pixel81',
              'mm2pixel1', 'mm2pixel2', 'mm2pixel3', 'mm2pixel4', 'mm2pixel5', 'mm2pixel6', 'mm2pixel7', 'mm2pixel8', 'mm2pixel9', 'mm2pixel10', 'mm2pixel11', 'mm2pixel12', 'mm2pixel13', 'mm2pixel14', 'mm2pixel15', 'mm2pixel16', 'mm2pixel17', 'mm2pixel18', 'mm2pixel19', 'mm2pixel20', 'mm2pixel21', 'mm2pixel22', 'mm2pixel23', 'mm2pixel24', 'mm2pixel25', 'mm2pixel26', 'mm2pixel27', 'mm2pixel28', 'mm2pixel29', 'mm2pixel30', 'mm2pixel31', 'mm2pixel32', 'mm2pixel33', 'mm2pixel34', 'mm2pixel35', 'mm2pixel36', 'mm2pixel37', 'mm2pixel38', 'mm2pixel39', 'mm2pixel40', 'mm2pixel41', 'mm2pixel42', 'mm2pixel43', 'mm2pixel44', 'mm2pixel45', 'mm2pixel46', 'mm2pixel47', 'mm2pixel48', 'mm2pixel49', 'mm2pixel50', 'mm2pixel51', 'mm2pixel52', 'mm2pixel53', 'mm2pixel54', 'mm2pixel55', 'mm2pixel56', 'mm2pixel57', 'mm2pixel58', 'mm2pixel59', 'mm2pixel60', 'mm2pixel61', 'mm2pixel62', 'mm2pixel63', 'mm2pixel64', 'mm2pixel65', 'mm2pixel66', 'mm2pixel67', 'mm2pixel68', 'mm2pixel69', 'mm2pixel70', 'mm2pixel71', 'mm2pixel72', 'mm2pixel73', 'mm2pixel74', 'mm2pixel75', 'mm2pixel76', 'mm2pixel77', 'mm2pixel78', 'mm2pixel79', 'mm2pixel80', 'mm2pixel81', 
              'mm3pixel1', 'mm3pixel2', 'mm3pixel3', 'mm3pixel4', 'mm3pixel5', 'mm3pixel6', 'mm3pixel7', 'mm3pixel8', 'mm3pixel9', 'mm3pixel10', 'mm3pixel11', 'mm3pixel12', 'mm3pixel13', 'mm3pixel14', 'mm3pixel15', 'mm3pixel16', 'mm3pixel17', 'mm3pixel18', 'mm3pixel19', 'mm3pixel20', 'mm3pixel21', 'mm3pixel22', 'mm3pixel23', 'mm3pixel24', 'mm3pixel25', 'mm3pixel26', 'mm3pixel27', 'mm3pixel28', 'mm3pixel29', 'mm3pixel30', 'mm3pixel31', 'mm3pixel32', 'mm3pixel33', 'mm3pixel34', 'mm3pixel35', 'mm3pixel36', 'mm3pixel37', 'mm3pixel38', 'mm3pixel39', 'mm3pixel40', 'mm3pixel41', 'mm3pixel42', 'mm3pixel43', 'mm3pixel44', 'mm3pixel45', 'mm3pixel46', 'mm3pixel47', 'mm3pixel48', 'mm3pixel49', 'mm3pixel50', 'mm3pixel51', 'mm3pixel52', 'mm3pixel53', 'mm3pixel54', 'mm3pixel55', 'mm3pixel56', 'mm3pixel57', 'mm3pixel58', 'mm3pixel59', 'mm3pixel60', 'mm3pixel61', 'mm3pixel62', 'mm3pixel63', 'mm3pixel64', 'mm3pixel65', 'mm3pixel66', 'mm3pixel67', 'mm3pixel68', 'mm3pixel69', 'mm3pixel70', 'mm3pixel71', 'mm3pixel72', 'mm3pixel73', 'mm3pixel74', 'mm3pixel75', 'mm3pixel76', 'mm3pixel77', 'mm3pixel78', 'mm3pixel79', 'mm3pixel80', 'mm3pixel81']


# lern_var = [ 'nuray_41', 'nuray_42', 'nuray_43', 'nuray_44', 'nuray_45', 'nuray_46', 'nuray_47', 'nuray_48', 'nuray_49', 'nuray_50'] 

lern_var = ['nuray_1', 'nuray_2', 'nuray_3', 'nuray_4', 'nuray_5', 'nuray_6', 'nuray_7', 'nuray_8', 'nuray_9', 'nuray_10', 'nuray_11', 'nuray_12', 'nuray_13', 'nuray_14', 'nuray_15', 'nuray_16', 'nuray_17', 'nuray_18', 'nuray_19', 'nuray_20', 'nuray_21', 'nuray_22', 'nuray_23', 'nuray_24', 'nuray_25', 'nuray_26', 'nuray_27', 'nuray_28', 'nuray_29', 'nuray_30', 'nuray_31', 'nuray_32', 'nuray_33', 'nuray_34', 'nuray_35', 'nuray_36', 'nuray_37', 'nuray_38', 'nuray_39', 'nuray_40', 'nuray_41', 'nuray_42', 'nuray_43', 'nuray_44', 'nuray_45', 'nuray_46', 'nuray_47', 'nuray_48', 'nuray_49', 'nuray_50'] 


# input_list = ['mm1pixel1', 'mm1pixel2', 'mm1pixel3', 'mm1pixel4', 'mm1pixel5', 'mm1pixel6', 'mm1pixel7', 'mm1pixel8', 'mm1pixel9', 'mm1pixel10', 'mm1pixel11', 'mm1pixel12', 'mm1pixel13', 'mm1pixel14', 'mm1pixel15', 'mm1pixel16', 'mm1pixel17', 'mm1pixel18', 'mm1pixel19', 'mm1pixel20', 'mm1pixel21', 'mm1pixel22', 'mm1pixel23', 'mm1pixel24', 'mm1pixel25', 'mm1pixel26', 'mm1pixel27', 'mm1pixel28', 'mm1pixel29', 'mm1pixel30', 'mm1pixel31', 'mm1pixel32', 'mm1pixel33', 'mm1pixel34', 'mm1pixel35', 'mm1pixel36', 'mm1pixel37', 'mm1pixel38', 'mm1pixel39', 'mm1pixel40', 'mm1pixel41', 'mm1pixel42', 'mm1pixel43', 'mm1pixel44', 'mm1pixel45', 'mm1pixel46', 'mm1pixel47', 'mm1pixel48', 'mm1pixel49', 'mm1pixel50', 'mm1pixel51', 'mm1pixel52', 'mm1pixel53', 'mm1pixel54', 'mm1pixel55', 'mm1pixel56', 'mm1pixel57', 'mm1pixel58', 'mm1pixel59', 'mm1pixel60', 'mm1pixel61', 'mm1pixel62', 'mm1pixel63', 'mm1pixel64', 'mm1pixel65', 'mm1pixel66', 'mm1pixel67', 'mm1pixel68', 'mm1pixel69', 'mm1pixel70', 'mm1pixel71', 'mm1pixel72', 'mm1pixel73', 'mm1pixel74', 'mm1pixel75', 'mm1pixel76', 'mm1pixel77', 'mm1pixel78', 'mm1pixel79', 'mm1pixel80', 'mm1pixel81']
              


# lern_var = ['nuray_1','nuray_5','nuray_10','nuray_20','nuray_40'] 

# lern_var = 'hornI(kA)'
#%%

data_test_x = df_data_2[input_list]/20000

data_test_y = df_data_2[lern_var]


#%%

predictions = model.predict(data_test_x)



#%%
xpoint=np.linspace((0.5)*0.2, (50+0.5)*0.2, 50)

testy=np.array(data_test_y)




#%%


x = [n for n in range(1,len(data_test_x)+1)]
#predictions = model.predict(X_test)
#testy = np.array(y_test)
plt.figure(figsize = (9,7))
plt.plot(x,predictions[:,9] , linewidth = 3)
plt.plot(x, testy[:,9], linewidth = 3)
plt.xlabel('index')
plt.ylabel('weighted events of neutrino at 2 GeV')
plt.legend(["Prediction", "MC"], fontsize = 15)
# plt.ylim(-2.1,2.1)
# plt.legend(["hptgt", "vptgt"], fontsize = 15)
plt.grid()
plt.tight_layout()



#%%







#%%
input_list_2 =['hptgt(cm)']
input_list_3 =['vptgt(cm)']
input_list_4 =['hornI(kA)']
xpos = df_data_2[input_list_2 ]
ypos = df_data_2[input_list_3 ]
hornI = df_data_2[input_list_4 ]


#%%
fig, axs = plt.subplots(2, 1, figsize=(7, 5), constrained_layout=True)

axs[0].plot(xpoint, predictions[i], linewidth=3)
axs[0].plot(xpoint, testy[i], "--", linewidth=3)
axs[0].set_ylabel('Normalized neutrino events')
axs[0].set_ylim(0, 0.9)
axs[0].legend(["Prediction", "MC"], fontsize=15)
axs[0].grid()

axs[1].plot(xpoint, ratios, drawstyle='steps-pre')
axs[1].set_xlabel('Neutrino energy at NOvA Near (GeV)')
axs[1].set_ylabel('Ratios')
axs[1].set_ylim(0.95, 1.05)
axs[1].grid()

plt.tight_layout()



#%%



i =66

ratios=predictions[i]/testy[i]

x = [n for n in range(1,i+1)]

#%%
fig = plt.figure(constrained_layout=True)
axs = fig.subplot_mosaic([['Left', 'Right'],['Left', 'Right_2'],['Left_2', 'Right_3']],
                          gridspec_kw={'width_ratios':[2, 1]})


# axs['Left'].set_title('Plot on Left')
axs['Left'].plot((xpoint), predictions[i], linewidth = 3)
axs['Left'].plot((xpoint), testy[i], "--", linewidth = 3)
# axs['Left'].set_xlabel('neutrinos energy at NOvA Near (GeV)')
axs['Left'].set_ylabel('Normlized neutrino events')
axs['Left'].set_ylim(0,0.9)
axs['Left'].legend(["Prediction", "MC-Data"], fontsize = 15)
axs['Left'].grid()


axs['Left_2'].plot((xpoint), ratios, drawstyle='steps-pre')
axs['Left_2'].set_xlabel('Neutrino energy at NOvA Near (GeV)')
axs['Left_2'].set_ylabel('Ratios')
axs['Left_2'].set_ylim(0.95,1.05)
axs['Left_2'].grid()
  

axs['Right'].plot(x, hornI[:i], 'r.')
# axs['Right_2'].set_xlabel('index')
axs['Right'].set_ylabel('HornI (kA)')
axs['Right'].set_ylim(175,205)
axs['Right'].set_xlim(0,201)
axs['Right'].grid()


# axs['Right_2'].set_title('Plot Top Right')
axs['Right_2'].plot(x, xpos[:i]*10, 'b.')
# axs['Right_2'].set_xlabel('index')
axs['Right_2'].set_ylabel('BeamX (mm)')
axs['Right_2'].set_ylim(-2.1,2.1)
axs['Right_2'].set_xlim(0,201)
axs['Right_2'].grid()




axs['Right_3'].plot(x, ypos[:i]*10, 'y.')
axs['Right_3'].set_xlabel('index')
axs['Right_3'].set_ylabel('BeamY (mm)')
axs['Right_3'].set_ylim(-2.1,2.1)
axs['Right_3'].set_xlim(0,201)
axs['Right_3'].grid()


 
#%%
for i in range(1, 300+1,5):
    
    ratios=predictions[i]/testy[i]

    x = [n for n in range(1,i+1)]
    
    fig = plt.figure(constrained_layout=True)
    axs = fig.subplot_mosaic([['Left', 'Right'],['Left', 'Right_2'],['Left_2', 'Right_3']],
                              gridspec_kw={'width_ratios':[2, 1]})


    # axs['Left'].set_title('Plot on Left')
    axs['Left'].plot((xpoint), predictions[i], linewidth = 3)
    axs['Left'].plot((xpoint), testy[i], "--", linewidth = 3)
    # axs['Left'].set_xlabel('neutrinos energy at NOvA Near (GeV)')
    axs['Left'].set_ylabel('Normalized neutrino events')
    axs['Left'].set_ylim(0,0.9)
    
    axs['Left'].legend(["Prediction", "MC"], fontsize = 15)
    axs['Left'].grid()


    axs['Left_2'].plot((xpoint), ratios, drawstyle='steps-pre')
    axs['Left_2'].set_xlabel('Neutrino energy at NOvA Near (GeV)')
    axs['Left_2'].set_ylabel('Ratios')
    axs['Left_2'].set_ylim(0.95,1.05)
    axs['Left_2'].grid()
      

    axs['Right'].plot(x, hornI[:i], 'r.')
    # axs['Right_2'].set_xlabel('index')
    axs['Right'].set_ylabel('Horn I (kA)')
    axs['Right'].set_ylim(175,205)
    axs['Right'].set_xlim(0,301)
    axs['Right'].grid()


    # axs['Right_2'].set_title('Plot Top Right')
    axs['Right_2'].plot(x, xpos[:i]*10, 'b.')
    # axs['Right_2'].set_xlabel('index')
    axs['Right_2'].set_ylabel('Beam X (mm)')
    axs['Right_2'].set_ylim(-2.1,2.1)
    axs['Right_2'].set_xlim(0,301)
    axs['Right_2'].grid()




    axs['Right_3'].plot(x, ypos[:i]*10, 'y.')
    axs['Right_3'].set_xlabel('index')
    axs['Right_3'].set_ylabel('Beam Y (mm)')
    axs['Right_3'].set_ylim(-2.1,2.1)
    axs['Right_3'].set_xlim(0,301)
    axs['Right_3'].grid()


    plt.savefig(f'/Users/yidingyu/npz/csv/plots/Nuray-{i}.png')
    plt.close()


#%%
with imageio.get_writer('/Users/yidingyu/npz/csv/plots/Nuray_latest.gif', mode='i') as writer:
    for i in range(1, 300+1,5):
        image = imageio.imread(f'/Users/yidingyu/npz/csv/plots/Nuray-{i}.png')
        writer.append_data(image)





#%%

ratios=testy[i]/testy[274]

for i in range(1, 300+1,5):
    
    ratios=predictions[i]/testy[i]

    x = [n for n in range(1,i+1)]
    
    fig = plt.figure(constrained_layout=True)
    axs = fig.subplot_mosaic([['Left', 'Right'],['Left', 'Right_2'],['Left_2', 'Right_3']],
                              gridspec_kw={'width_ratios':[2, 1]})


    # axs['Left'].set_title('Plot on Left')
    #axs['Left'].plot((xpoint), predictions[i], linewidth = 3)
    axs['Left'].plot((xpoint), testy[274], "--", linewidth = 3)
    axs['Left'].plot((xpoint), testy[i], "--", linewidth = 3)
    
    # axs['Left'].set_xlabel('neutrinos energy at NOvA Near (GeV)')
    axs['Left'].set_ylabel('Normlized neutrino events')
    axs['Left'].set_ylim(0,0.9)
    
    axs['Left'].legend(["Nominal", "MC-Data"], fontsize = 15)
    axs['Left'].grid()


    axs['Left_2'].plot((xpoint), ratios, drawstyle='steps-pre')
    axs['Left_2'].set_xlabel('Neutrino energy at NOvA Near (GeV)')
    axs['Left_2'].set_ylabel('Ratios')
    axs['Left_2'].set_ylim(0.95,1.05)
    axs['Left_2'].grid()
      

    axs['Right'].plot(x, hornI[:i], 'r.')
    # axs['Right_2'].set_xlabel('index')
    axs['Right'].set_ylabel('HornI (kA)')
    axs['Right'].set_ylim(175,205)
    axs['Right'].set_xlim(0,301)
    axs['Right'].grid()


    # axs['Right_2'].set_title('Plot Top Right')
    axs['Right_2'].plot(x, xpos[:i]*10, 'b.')
    # axs['Right_2'].set_xlabel('index')
    axs['Right_2'].set_ylabel('BeamX (mm)')
    axs['Right_2'].set_ylim(-2.1,2.1)
    axs['Right_2'].set_xlim(0,301)
    axs['Right_2'].grid()




    axs['Right_3'].plot(x, ypos[:i]*10, 'y.')
    axs['Right_3'].set_xlabel('index')
    axs['Right_3'].set_ylabel('BeamY (mm)')
    axs['Right_3'].set_ylim(-2.1,2.1)
    axs['Right_3'].set_xlim(0,301)
    axs['Right_3'].grid()


    plt.savefig(f'/Users/yidingyu/npz/csv/plots/Nuray-{i}.png')
    plt.close()


#%%
with imageio.get_writer('/Users/yidingyu/npz/csv/plots/Nuray_check.gif', mode='i') as writer:
    for i in range(1, 300+1,5):
        image = imageio.imread(f'/Users/yidingyu/npz/csv/plots/Nuray-{i}.png')
        writer.append_data(image)


