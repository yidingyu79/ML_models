#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:53:13 2024

@author: yidingyu
"""

import pandas as pd
import numpy as np
from scipy import optimize
import csv
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#------------------
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

#==============
#checking data:
#==============



df_data.reset_index().plot(x = 'index',y = ['hornI(kA)'],markersize = 1, marker='o', linestyle='')
plt.xlabel('index')
#plt.figure(constrained_layout=True)
plt.grid(True)
plt.ylabel('Beam Position [cm]')
#plt.savefig("plots/test_beamX.png");
plt.show()
#%%
'''
df_data.reset_index().plot(x = 'index',y = ['hptgt(cm)','vptgt(cm)'],markersize = 1, marker='o', linestyle='')
plt.xlabel('index')
#plt.figure(constrained_layout=True)
plt.grid(True)
plt.ylabel('Beam Position [cm]')
plt.savefig("plots/test_beamX.png");
plt.show()


df_data.reset_index().plot(x = 'index',y = ['mm1pixel1','mm1pixel2'],markersize = 1, marker='o', linestyle='')
plt.xlabel('index')
#plt.figure(constrained_layout=True)
plt.grid(True)
plt.ylabel('MM1 # events')
plt.savefig("plots/test_pix.png");
plt.show()


df_data.plot(x = 'vptgt(cm)',y =['mm1pixel63','mm1pixel73'],markersize = 2, marker='o', linestyle='')
#plt.ylim((0.85, 1.0))
plt.ylabel('Number of events')
plt.xlabel('Beam Position')
plt.grid(True)
plt.savefig('plots/mmyav_Xscan_beamY_corr.png')
plt.show()
'''
#%%
#=====================
input_list = ['mm1pixel1', 'mm1pixel2', 'mm1pixel3', 'mm1pixel4', 'mm1pixel5', 'mm1pixel6', 'mm1pixel7', 'mm1pixel8', 'mm1pixel9', 'mm1pixel10', 'mm1pixel11', 'mm1pixel12', 'mm1pixel13', 'mm1pixel14', 'mm1pixel15', 'mm1pixel16', 'mm1pixel17', 'mm1pixel18', 'mm1pixel19', 'mm1pixel20', 'mm1pixel21', 'mm1pixel22', 'mm1pixel23', 'mm1pixel24', 'mm1pixel25', 'mm1pixel26', 'mm1pixel27', 'mm1pixel28', 'mm1pixel29', 'mm1pixel30', 'mm1pixel31', 'mm1pixel32', 'mm1pixel33', 'mm1pixel34', 'mm1pixel35', 'mm1pixel36', 'mm1pixel37', 'mm1pixel38', 'mm1pixel39', 'mm1pixel40', 'mm1pixel41', 'mm1pixel42', 'mm1pixel43', 'mm1pixel44', 'mm1pixel45', 'mm1pixel46', 'mm1pixel47', 'mm1pixel48', 'mm1pixel49', 'mm1pixel50', 'mm1pixel51', 'mm1pixel52', 'mm1pixel53', 'mm1pixel54', 'mm1pixel55', 'mm1pixel56', 'mm1pixel57', 'mm1pixel58', 'mm1pixel59', 'mm1pixel60', 'mm1pixel61', 'mm1pixel62', 'mm1pixel63', 'mm1pixel64', 'mm1pixel65', 'mm1pixel66', 'mm1pixel67', 'mm1pixel68', 'mm1pixel69', 'mm1pixel70', 'mm1pixel71', 'mm1pixel72', 'mm1pixel73', 'mm1pixel74', 'mm1pixel75', 'mm1pixel76', 'mm1pixel77', 'mm1pixel78', 'mm1pixel79', 'mm1pixel80', 'mm1pixel81',
              'mm2pixel1', 'mm2pixel2', 'mm2pixel3', 'mm2pixel4', 'mm2pixel5', 'mm2pixel6', 'mm2pixel7', 'mm2pixel8', 'mm2pixel9', 'mm2pixel10', 'mm2pixel11', 'mm2pixel12', 'mm2pixel13', 'mm2pixel14', 'mm2pixel15', 'mm2pixel16', 'mm2pixel17', 'mm2pixel18', 'mm2pixel19', 'mm2pixel20', 'mm2pixel21', 'mm2pixel22', 'mm2pixel23', 'mm2pixel24', 'mm2pixel25', 'mm2pixel26', 'mm2pixel27', 'mm2pixel28', 'mm2pixel29', 'mm2pixel30', 'mm2pixel31', 'mm2pixel32', 'mm2pixel33', 'mm2pixel34', 'mm2pixel35', 'mm2pixel36', 'mm2pixel37', 'mm2pixel38', 'mm2pixel39', 'mm2pixel40', 'mm2pixel41', 'mm2pixel42', 'mm2pixel43', 'mm2pixel44', 'mm2pixel45', 'mm2pixel46', 'mm2pixel47', 'mm2pixel48', 'mm2pixel49', 'mm2pixel50', 'mm2pixel51', 'mm2pixel52', 'mm2pixel53', 'mm2pixel54', 'mm2pixel55', 'mm2pixel56', 'mm2pixel57', 'mm2pixel58', 'mm2pixel59', 'mm2pixel60', 'mm2pixel61', 'mm2pixel62', 'mm2pixel63', 'mm2pixel64', 'mm2pixel65', 'mm2pixel66', 'mm2pixel67', 'mm2pixel68', 'mm2pixel69', 'mm2pixel70', 'mm2pixel71', 'mm2pixel72', 'mm2pixel73', 'mm2pixel74', 'mm2pixel75', 'mm2pixel76', 'mm2pixel77', 'mm2pixel78', 'mm2pixel79', 'mm2pixel80', 'mm2pixel81', 
              'mm3pixel1', 'mm3pixel2', 'mm3pixel3', 'mm3pixel4', 'mm3pixel5', 'mm3pixel6', 'mm3pixel7', 'mm3pixel8', 'mm3pixel9', 'mm3pixel10', 'mm3pixel11', 'mm3pixel12', 'mm3pixel13', 'mm3pixel14', 'mm3pixel15', 'mm3pixel16', 'mm3pixel17', 'mm3pixel18', 'mm3pixel19', 'mm3pixel20', 'mm3pixel21', 'mm3pixel22', 'mm3pixel23', 'mm3pixel24', 'mm3pixel25', 'mm3pixel26', 'mm3pixel27', 'mm3pixel28', 'mm3pixel29', 'mm3pixel30', 'mm3pixel31', 'mm3pixel32', 'mm3pixel33', 'mm3pixel34', 'mm3pixel35', 'mm3pixel36', 'mm3pixel37', 'mm3pixel38', 'mm3pixel39', 'mm3pixel40', 'mm3pixel41', 'mm3pixel42', 'mm3pixel43', 'mm3pixel44', 'mm3pixel45', 'mm3pixel46', 'mm3pixel47', 'mm3pixel48', 'mm3pixel49', 'mm3pixel50', 'mm3pixel51', 'mm3pixel52', 'mm3pixel53', 'mm3pixel54', 'mm3pixel55', 'mm3pixel56', 'mm3pixel57', 'mm3pixel58', 'mm3pixel59', 'mm3pixel60', 'mm3pixel61', 'mm3pixel62', 'mm3pixel63', 'mm3pixel64', 'mm3pixel65', 'mm3pixel66', 'mm3pixel67', 'mm3pixel68', 'mm3pixel69', 'mm3pixel70', 'mm3pixel71', 'mm3pixel72', 'mm3pixel73', 'mm3pixel74', 'mm3pixel75', 'mm3pixel76', 'mm3pixel77', 'mm3pixel78', 'mm3pixel79', 'mm3pixel80', 'mm3pixel81']
lern_var = 'hptgt(cm)'

X_train, X_test, y_train, y_test = train_test_split(df_data[input_list],df_data[lern_var],test_size=0.3, shuffle = True)

print(X_train)
print(y_train)

train_err, test_err = [],[]

## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
##https://www.tutorialspoint.com/scikit_learn/scikit_learn_ridge_regression.htm
#model = Ridge(alpha=.001)

#%%
model = LinearRegression()
model.fit(X_train,y_train)
#model = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)

# for m in range(1,len(X_train)):
#     model.fit(X_train[:m],y_train[:m])
#     y_train_pred = model.predict(X_train[:m])
#     y_test_pred = model.predict(X_test[:m])
#     train_err.append(mean_squared_error(y_train_pred, y_train[:m]))
#     test_err.append(mean_squared_error(y_test_pred, y_test[:m]))
#     if(m%500 == 0):
#         print(m,mean_squared_error(y_train_pred, y_train[:m]))
#     #== Optional:
#     if abs(mean_squared_error(y_train_pred, y_train[:m]) - mean_squared_error(y_test_pred, y_test[:m])) < 0.0000001:
#         break



# #%%
# #model.fit(X_train,y_train)
# plt.figure(1)
# plt.figure(constrained_layout=True)
# plt.plot(np.sqrt(train_err),"b-.",linewidth = 2, label = "Train")
# plt.plot(np.sqrt(test_err),"r--",linewidth = 2, label = "Validation")
# plt.legend(loc='best')
# plt.ylabel('Error (RMS)')
# plt.xlabel('Training set size')
# plt.yscale('log')
# plt.grid(True)
# # plt.savefig("plots/model_rms.png")
# plt.show()



#%%
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Training set score: {:.3f}".format(model.score(X_train, y_train)))
print("Test set score: {:.3f}".format(model.score(X_test, y_test)))
print("Parameters: ", model.coef_, model.intercept_ )


#%%
#---  Save model:
filename = '/Users/yidingyu/npz/csv/hptgt_model1.pkl'
pickle.dump(model, open(filename, 'wb'))



index = [i for i in range(len(y_test))]
    
df_new = pd.DataFrame({'index':index,'ytest':y_test,'ypred':y_pred})

df_new['diff'] = df_new.ytest - df_new.ypred

plt.hist(df_new['diff'],bins =100,facecolor='blue', alpha=0.7,edgecolor = 'k')
plt.xlabel('diff (kA)')
# plt.savefig("plots/diff.png");
plt.show()


#%%
df_new['diff2'] = (df_new['ytest'] - df_new['ypred'])**2
print(df_new)

#--- RMS:
sdv = np.sqrt(df_new['diff2'].sum()/(len(df_new.diff2)-1))
print(len(df_new.diff2)-1)
print('Standard Div: ',sdv)



#%%
#---------------------
## Testing saved model
#---------------------
#input_list = ['mm1pix1','mm1pix8','mm1pix9','mm1pix10','mm1pix45','mm1pix46','mm1pix53','mm1pix54','mm1pix55','mm1pix56','mm1pix62','mm1pix63','mm1pix64','mm1pix65','mm1pix73', 'mm2pix0','mm2pix1','mm2pix7','mm2pix8','mm2pix9','mm2pix10','mm2pix17','mm2pix18','mm2pix62','mm2pix63','mm2pix64','mm2pix70','mm2pix71','mm2pix72','mm2pix73','mm2pix79','mm2pix80','mm1cor','mm2cor']

filename = '/Users/yidingyu/npz/csv/hptgt_model1.pkl'
model1 = pickle.load(open(filename, 'rb'))

pred_t = model1.predict(df_data[input_list])

df_data['pred'] = pred_t
print(df_data)

df_data.reset_index().plot(x = 'index',y = [str(lern_var),'pred'],markersize = 3, marker='o', linestyle='--')
plt.ylabel('pred')
#plt.figure(constrained_layout=True)
plt.grid(True)
plt.xlabel('index')
#plt.savefig("plots/test_hornI_pred.png");
plt.show()
#%%
filename = '/Users/yidingyu/npz/csv/hptgt_model1.pkl'
model1 = pickle.load(open(filename, 'rb'))


d2_200 = pd.read_csv('/Users/yidingyu/npz/csv/nuray_test_50.csv',sep = ',')

df_data_2 = d2_200 #pd.concat([dtg7,dtg8,dtg9,dtg10,df1,df2,df3],ignore_index = True)
print(df_data_2)
df_data_2=df_data_2.dropna()


# input_list = ['mm1pixel1', 'mm1pixel2', 'mm1pixel3', 'mm1pixel4', 'mm1pixel5', 'mm1pixel6', 'mm1pixel7', 'mm1pixel8', 'mm1pixel9', 'mm1pixel10', 'mm1pixel11', 'mm1pixel12', 'mm1pixel13', 'mm1pixel14', 'mm1pixel15', 'mm1pixel16', 'mm1pixel17', 'mm1pixel18', 'mm1pixel19', 'mm1pixel20', 'mm1pixel21', 'mm1pixel22', 'mm1pixel23', 'mm1pixel24', 'mm1pixel25', 'mm1pixel26', 'mm1pixel27', 'mm1pixel28', 'mm1pixel29', 'mm1pixel30', 'mm1pixel31', 'mm1pixel32', 'mm1pixel33', 'mm1pixel34', 'mm1pixel35', 'mm1pixel36', 'mm1pixel37', 'mm1pixel38', 'mm1pixel39', 'mm1pixel40', 'mm1pixel41', 'mm1pixel42', 'mm1pixel43', 'mm1pixel44', 'mm1pixel45', 'mm1pixel46', 'mm1pixel47', 'mm1pixel48', 'mm1pixel49', 'mm1pixel50', 'mm1pixel51', 'mm1pixel52', 'mm1pixel53', 'mm1pixel54', 'mm1pixel55', 'mm1pixel56', 'mm1pixel57', 'mm1pixel58', 'mm1pixel59', 'mm1pixel60', 'mm1pixel61', 'mm1pixel62', 'mm1pixel63', 'mm1pixel64', 'mm1pixel65', 'mm1pixel66', 'mm1pixel67', 'mm1pixel68', 'mm1pixel69', 'mm1pixel70', 'mm1pixel71', 'mm1pixel72', 'mm1pixel73', 'mm1pixel74', 'mm1pixel75', 'mm1pixel76', 'mm1pixel77', 'mm1pixel78', 'mm1pixel79', 'mm1pixel80', 'mm1pixel81']
  

input_list = ['mm1pixel1', 'mm1pixel2', 'mm1pixel3', 'mm1pixel4', 'mm1pixel5', 'mm1pixel6', 'mm1pixel7', 'mm1pixel8', 'mm1pixel9', 'mm1pixel10', 'mm1pixel11', 'mm1pixel12', 'mm1pixel13', 'mm1pixel14', 'mm1pixel15', 'mm1pixel16', 'mm1pixel17', 'mm1pixel18', 'mm1pixel19', 'mm1pixel20', 'mm1pixel21', 'mm1pixel22', 'mm1pixel23', 'mm1pixel24', 'mm1pixel25', 'mm1pixel26', 'mm1pixel27', 'mm1pixel28', 'mm1pixel29', 'mm1pixel30', 'mm1pixel31', 'mm1pixel32', 'mm1pixel33', 'mm1pixel34', 'mm1pixel35', 'mm1pixel36', 'mm1pixel37', 'mm1pixel38', 'mm1pixel39', 'mm1pixel40', 'mm1pixel41', 'mm1pixel42', 'mm1pixel43', 'mm1pixel44', 'mm1pixel45', 'mm1pixel46', 'mm1pixel47', 'mm1pixel48', 'mm1pixel49', 'mm1pixel50', 'mm1pixel51', 'mm1pixel52', 'mm1pixel53', 'mm1pixel54', 'mm1pixel55', 'mm1pixel56', 'mm1pixel57', 'mm1pixel58', 'mm1pixel59', 'mm1pixel60', 'mm1pixel61', 'mm1pixel62', 'mm1pixel63', 'mm1pixel64', 'mm1pixel65', 'mm1pixel66', 'mm1pixel67', 'mm1pixel68', 'mm1pixel69', 'mm1pixel70', 'mm1pixel71', 'mm1pixel72', 'mm1pixel73', 'mm1pixel74', 'mm1pixel75', 'mm1pixel76', 'mm1pixel77', 'mm1pixel78', 'mm1pixel79', 'mm1pixel80', 'mm1pixel81',
              'mm2pixel1', 'mm2pixel2', 'mm2pixel3', 'mm2pixel4', 'mm2pixel5', 'mm2pixel6', 'mm2pixel7', 'mm2pixel8', 'mm2pixel9', 'mm2pixel10', 'mm2pixel11', 'mm2pixel12', 'mm2pixel13', 'mm2pixel14', 'mm2pixel15', 'mm2pixel16', 'mm2pixel17', 'mm2pixel18', 'mm2pixel19', 'mm2pixel20', 'mm2pixel21', 'mm2pixel22', 'mm2pixel23', 'mm2pixel24', 'mm2pixel25', 'mm2pixel26', 'mm2pixel27', 'mm2pixel28', 'mm2pixel29', 'mm2pixel30', 'mm2pixel31', 'mm2pixel32', 'mm2pixel33', 'mm2pixel34', 'mm2pixel35', 'mm2pixel36', 'mm2pixel37', 'mm2pixel38', 'mm2pixel39', 'mm2pixel40', 'mm2pixel41', 'mm2pixel42', 'mm2pixel43', 'mm2pixel44', 'mm2pixel45', 'mm2pixel46', 'mm2pixel47', 'mm2pixel48', 'mm2pixel49', 'mm2pixel50', 'mm2pixel51', 'mm2pixel52', 'mm2pixel53', 'mm2pixel54', 'mm2pixel55', 'mm2pixel56', 'mm2pixel57', 'mm2pixel58', 'mm2pixel59', 'mm2pixel60', 'mm2pixel61', 'mm2pixel62', 'mm2pixel63', 'mm2pixel64', 'mm2pixel65', 'mm2pixel66', 'mm2pixel67', 'mm2pixel68', 'mm2pixel69', 'mm2pixel70', 'mm2pixel71', 'mm2pixel72', 'mm2pixel73', 'mm2pixel74', 'mm2pixel75', 'mm2pixel76', 'mm2pixel77', 'mm2pixel78', 'mm2pixel79', 'mm2pixel80', 'mm2pixel81', 
              'mm3pixel1', 'mm3pixel2', 'mm3pixel3', 'mm3pixel4', 'mm3pixel5', 'mm3pixel6', 'mm3pixel7', 'mm3pixel8', 'mm3pixel9', 'mm3pixel10', 'mm3pixel11', 'mm3pixel12', 'mm3pixel13', 'mm3pixel14', 'mm3pixel15', 'mm3pixel16', 'mm3pixel17', 'mm3pixel18', 'mm3pixel19', 'mm3pixel20', 'mm3pixel21', 'mm3pixel22', 'mm3pixel23', 'mm3pixel24', 'mm3pixel25', 'mm3pixel26', 'mm3pixel27', 'mm3pixel28', 'mm3pixel29', 'mm3pixel30', 'mm3pixel31', 'mm3pixel32', 'mm3pixel33', 'mm3pixel34', 'mm3pixel35', 'mm3pixel36', 'mm3pixel37', 'mm3pixel38', 'mm3pixel39', 'mm3pixel40', 'mm3pixel41', 'mm3pixel42', 'mm3pixel43', 'mm3pixel44', 'mm3pixel45', 'mm3pixel46', 'mm3pixel47', 'mm3pixel48', 'mm3pixel49', 'mm3pixel50', 'mm3pixel51', 'mm3pixel52', 'mm3pixel53', 'mm3pixel54', 'mm3pixel55', 'mm3pixel56', 'mm3pixel57', 'mm3pixel58', 'mm3pixel59', 'mm3pixel60', 'mm3pixel61', 'mm3pixel62', 'mm3pixel63', 'mm3pixel64', 'mm3pixel65', 'mm3pixel66', 'mm3pixel67', 'mm3pixel68', 'mm3pixel69', 'mm3pixel70', 'mm3pixel71', 'mm3pixel72', 'mm3pixel73', 'mm3pixel74', 'mm3pixel75', 'mm3pixel76', 'mm3pixel77', 'mm3pixel78', 'mm3pixel79', 'mm3pixel80', 'mm3pixel81']



lern_var = ['hptgt(cm)']


data_test_x = df_data_2[input_list]

data_test_y = df_data_2[lern_var]

predictions = model.predict(data_test_x)

x = [n for n in range(1,len(data_test_x)+1)]

#%%
testy=np.array(data_test_y)
testy = testy.flatten()


difference = predictions - testy

plt.figure(figsize=(9, 7))
plt.figure(figsize=(9, 7))
plt.scatter(x, predictions, marker='*', s=15, label='Prediction')
plt.scatter(x, data_test_y, marker='+', s=15, label='MC')
plt.xlabel('Index')
plt.ylabel('Proton beam x (cm)')
plt.legend(fontsize=15)
plt.tight_layout() 
# plt.ylim(-2.1, 2.1)
# plt.legend(["hptgt", "vptgt"], fontsize=15)
plt.grid()
#%%
# Calculate and display RMS of the difference
rms_difference = np.sqrt(np.mean(np.square(difference)))
print(f'RMS of the Difference: {rms_difference}')

# Make a plot showing the RMS of the difference
plt.figure(figsize=(9, 7))
plt.scatter(x, difference, marker='o', s=10)
plt.axhline(y=rms_difference, color='red', linestyle='--', label=f'RMS: {rms_difference:.2f}')
plt.xlabel('Index')
plt.ylabel('Difference (Prediction - MC)')
plt.legend()
plt.grid()
#%%
plt.figure(figsize=(9, 7))
plt.hist(difference, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Difference (Prediction - MC) (cm)')
plt.ylabel('Frequency')
plt.title(f'RMS of difference: {rms_difference:.5f} cm')
plt
plt.grid()


#%%
        # Create subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Subplot 1: Scatter plot
axs[0].scatter(x, predictions, marker='*', s=15, label='Prediction')
axs[0].scatter(x, data_test_y, marker='+', s=15, label='MC')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Proton beam x (cm)')
axs[0].legend(fontsize=15)
axs[0].grid()

# Subplot 2: Histogram
axs[1].hist(difference, bins=50, edgecolor='black', alpha=0.7)
axs[1].set_xlabel('Difference (Prediction - MC) (cm)')
axs[1].set_ylabel('Frequency')
axs[1].set_title(f'RMS of difference: {rms_difference:.5f} cm')
axs[1].grid()

plt.tight_layout()  
plt.savefig('/Users/yidingyu/Documents/paper/ML_hptgt.pdf')

plt.show()
