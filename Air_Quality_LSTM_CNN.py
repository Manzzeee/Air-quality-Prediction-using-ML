#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/pk2971/Air-Quality-time-series/blob/main/Air_Quality_vs_Temperature_time_series_forecasting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Air Quality vs Temperature Time Series Forecasting.**
# 
# 
# Author: Praharshita Kaithepalli(pk2971@rit.edu)
# 
# 
# Data Set:https://archive.ics.uci.edu/ml/datasets/Air+Quality 

# Pollutants/Greenhouse gasses in the air often cause a rise in the temperature at any given time of the day. 
# As the day progresses and the traffic increases during a certain time of the day there might be more pollutants in the air(for example during the morning rush hours or when everyone is going back to homes or during the factorys working hours). Our goal is to see if we can predict the amount of pollutants in the air, temperature by studying the patterns across different times.
# 

# In[179]:


import pandas as pd
import io
import csv


# In[180]:


#from google.colab import drive
#drive.mount('/gdrive',force_remount=True)
#%cd /gdrive


# 

# In[181]:


df=pd.read_csv("AirQualityUCI.csv",sep=';',decimal='.',quoting=csv.QUOTE_NONE, skip_blank_lines=False)


# In[182]:


import pandas as pd
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense,Dropout
from tensorflow.keras.losses import MeanSquaredError as mse
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# In[183]:


#Dropping unnamed columns
df=df.drop(['Unnamed: 15','Unnamed: 16'],axis=1)


# In the data set date and time are in seperate columns. We will convert them into datetime data type and then concatenate them.

# In[184]:


from datetime import datetime as dt
df['Time'] = pd.to_datetime(df['Time'], format='%H.%M.%S' ).dt.time
df['Date']=pd.to_datetime(df['Date'],format='%d/%m/%Y')


# In[185]:


df['DateTime']=pd.to_datetime(df.Date.astype(str) + ' ' + df.Time.astype(str),errors='coerce')


# Some columns which are supposed to have float data type are in object data type and there is a "," instead of the decimal point.

# In[186]:


df['CO(GT)']=df['CO(GT)'].str.replace(",",".")
df['CO(GT)']=df['CO(GT)'].astype(float)


# In[187]:


df['C6H6(GT)']=df['C6H6(GT)'].str.replace(",",".")
df['C6H6(GT)']=df['C6H6(GT)'].astype(float)


# In[188]:


df['T']=df['T'].str.replace(",",".")
df['T']=df['T'].astype(float)


# In[189]:


df['RH']=df['RH'].str.replace(",",".")
df['RH']=df['RH'].astype(float)


# In[190]:


df['AH']=df['AH'].str.replace(",",".")
df['AH']=df['AH'].astype(float)


# In[191]:


df=df.dropna()


# **Exploratory Data Analysis:**
# 
# Performing EDA to understand how the data is distributed across Date and time. 

# In[192]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(df['DateTime'],df['CO(GT)'],color='blue')
ax.set(xlabel="Date",
       ylabel="Concentration of Carbon mono Oxide in the atmosphere",
       title="Concentration of CO by the date")

plt.show()


# Intersting observations here. There seems to be outliers in the data. Especially the value -200. Checking one more column to find out if there is similar outlier data.

# In[193]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(df['DateTime'],df['NO2(GT)'],color='blue')
ax.set(xlabel="Date",
       ylabel="Concentration of Nitrogen di Oxide in the atmosphere",
       title="Concentration of NO vs date")

plt.show()


# The outlier data is present in this column as well. 
# 
# Checking if there is outlier data specifically -200 in all the other columns.
# 

# In[194]:


print((df['CO(GT)']==-200).sum())


# In[195]:


#removing fields with -200 as value.
df=df[df['CO(GT)']!=-200]


# In[196]:


#plotting the data points again after cleaning.
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(df['DateTime'],df['CO(GT)'],color='blue')
ax.set(xlabel="Date",
       ylabel="Concentration of Carbon mono Oxide in the atmosphere",
       title="Concentration of CO by the date")

plt.show()


# We can see that the outliers have been removed. 

# In[197]:


print ("PT08.S1(CO) ",(df['PT08.S1(CO)']==-200).sum())
print("NMHC(GT) ",(df['NMHC(GT)']==-200).sum())
print("C6H6(GT) ",(df['C6H6(GT)']==-200).sum())
print("PT08.S2(NMHC)",(df['PT08.S2(NMHC)']==-200).sum())
print("NO2(GT)",(df['NO2(GT)']==-200).sum())
print("PT08.S4(NO2)",(df['PT08.S4(NO2)']==-200).sum())
print("PT08.S5(O3)",(df['PT08.S5(O3)']==-200).sum())
print("T",(df['T']==-200).sum())
print("RH",(df['RH']==-200).sum())
print("AH",(df['AH']==-200).sum())


# In[198]:


del df['NMHC(GT)']
#Too many outlier values we will drop the whole column as there is no use fixing the outlier.


# Removing the outlier and replacing it with mean/median value in the rest of the columns.

# In[199]:


median=df.loc[df['PT08.S1(CO)']!=-200,'PT08.S1(CO)'].median()
df.loc[df['PT08.S1(CO)']==-200,'PT08.S1(CO)']=np.nan
df['PT08.S1(CO)'].fillna(median,inplace=True)
median


# In[200]:


fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(df['DateTime'],df['PT08.S1(CO)'],color='blue')
ax.set(xlabel="Date",
       ylabel="Concentration of Carbon mono Oxide in the atmosphere",
       title="Concentration of CO by the date")

plt.show()


# In[201]:


median=df.loc[df['C6H6(GT)']!=-200,'C6H6(GT)'].median()
df.loc[df['C6H6(GT)']==-200,'C6H6(GT)']=np.nan
df['C6H6(GT)'].fillna(median,inplace=True)
median


# In[202]:


median=df.loc[df['PT08.S2(NMHC)']!=-200,'PT08.S2(NMHC)'].median()
df.loc[df['PT08.S2(NMHC)']==-200,'PT08.S2(NMHC)']=np.nan
df['PT08.S2(NMHC)'].fillna(median,inplace=True)
median


# In[203]:


median=df.loc[df['NOx(GT)']!=-200,'NOx(GT)'].median()
df.loc[df['NOx(GT)']==-200,'NOx(GT)']=np.nan
df['NOx(GT)'].fillna(median,inplace=True)
median


# In[204]:


median=df.loc[df['PT08.S3(NOx)']!=-200,'PT08.S3(NOx)'].median()
df.loc[df['PT08.S3(NOx)']==-200,'PT08.S3(NOx)']=np.nan
df['PT08.S3(NOx)'].fillna(median,inplace=True)
median


# In[205]:


median=df.loc[df['NO2(GT)']!=-200,'NO2(GT)'].median()
df.loc[df['NO2(GT)']==-200,'NO2(GT)']=np.nan
df['NO2(GT)'].fillna(median,inplace=True)
median


# In[206]:


median=df.loc[df['PT08.S4(NO2)']!=-200,'PT08.S4(NO2)'].median()
df.loc[df['PT08.S4(NO2)']==-200,'PT08.S4(NO2)']=np.nan
df['PT08.S4(NO2)'].fillna(median,inplace=True)
median


# In[207]:


median=df.loc[df['PT08.S5(O3)']!=-200,'PT08.S5(O3)'].median()
df.loc[df['PT08.S5(O3)']==-200,'PT08.S5(O3)']=np.nan
df['PT08.S5(O3)'].fillna(median,inplace=True)
median


# In[208]:


mean=df.loc[df['T']!=-200,'T'].mean()
df.loc[df['T']==-200,'T']=np.nan
df['T'].fillna(mean,inplace=True)
mean


# In[209]:


mean=df.loc[df['AH']!=-200,'AH'].mean()
df.loc[df['AH']==-200,'AH']=np.nan
df['AH'].fillna(mean,inplace=True)
mean


# In[210]:


mean=df.loc[df['RH']!=-200,'RH'].mean()
df.loc[df['RH']==-200,'RH']=np.nan
df['RH'].fillna(mean,inplace=True)
mean


# In[211]:


df.info()


# In[212]:


fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(df['DateTime'],df['PT08.S1(CO)'],color='blue')
ax.set(xlabel="Date",
       ylabel="Concentration of Carbon mono Oxide in the atmosphere",
       title="Concentration of CO by the date")

plt.show()


# In[213]:


df=df.dropna()


# In[214]:


df.plot(x='DateTime',y='T',figsize=(15,12))


# Converting the DateTime column into the index of the data set to better enable it for a time series forecasting.

# In[215]:


df.index=df.pop('DateTime')


# In[216]:


#Plotting the temperature against time is easier now.
temp=df['T']
temp.plot()


# In[217]:


temp.head(15)


# In[218]:


df.info()


# Now we get to the time series prediction. We split the Temperature column.
# 
# For example: 
# 
# t[1] t[2] t[3] t[4] t[5]-->t[6](t[6] is predicted by the first 5 data points) 
# 
# t[6] t[7] t[8] t[9] t[10]-->t[11]
# 
# t[11] t[12] t[13] t[14] t[15]-->t[16]
# 

# In[219]:


def df_to_X_y(df,window_size=5):
  df_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_np)-window_size):
    row = [[a] for a in df_np[i:i+window_size]]
    X.append(row)
    label = df_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)


# In[220]:


WINDOW_SIZE = 5
X1, y1 = df_to_X_y(temp, WINDOW_SIZE)
X1.shape, y1.shape


# Splitting the data into training, testing and validation sets.

# In[221]:


X_train1, y_train1 = X1[:5000], y1[:5000]
X_val1, y_val1 = X1[5000:5500], y1[5000:5500]
X_test1, y_test1 = X1[5500:], y1[5500:]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape


# Model 1: LSTM

# In[222]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

model1 = Sequential()
model1.add(InputLayer((5, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()


# In[223]:


#Optimizer=Adam
cp1 = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])


# In[224]:


model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=15, callbacks=[cp1])


# In[225]:


from tensorflow.keras.models import load_model
model1 = load_model('model1/')


# Model 1:Predicted data using X_train1 vs data in y_train1.
# 
# 
# 

# In[226]:


train_predictions = model1.predict(X_train1).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
train_results


# In[227]:


import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][50:100])
plt.plot(train_results['Actuals'][50:100])


# Model 2:Predicted data using validation data set X_val1 vs data in y_val1.
# 

# In[228]:


val_predictions = model1.predict(X_val1).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val1})
val_results


# In[229]:


import matplotlib.pyplot as plt
plt.plot(val_results['Val Predictions'][50:100])
plt.plot(val_results['Actuals'][50:100])


# Model 1:Predicted data using X_test1 vs data in y_test1.
# 
# 
# 
# 

# In[230]:


test_predictions = model1.predict(X_test1).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test1})
test_results


# In[231]:


import matplotlib.pyplot as plt
plt.plot(test_results['Test Predictions'][0:100])
plt.plot(test_results['Actuals'][0:100])


# In[232]:


#print("Mean Squared Error for LSTM model on test data:",losses.MeanSquaredError(y_test1, test_predictions))


# Model 2: Convolutional Neural Networks

# In[233]:


model2 = Sequential()
model2.add(InputLayer((5, 1)))
model2.add(Conv1D(64, kernel_size=2))
model2.add(Flatten())
model2.add(Dense(8, 'relu'))
model2.add(Dense(1, 'linear'))

model2.summary()


# In[234]:


#optimizer= Adam
cp2 = ModelCheckpoint('model2/', save_best_only=True)
model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])


# In[235]:


model2.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=15, callbacks=[cp2])


# In[236]:


from tensorflow.keras.models import load_model
model1 = load_model('model2/')


# Model 2:Predicted data using X_train1 vs data in y_train1.
# 

# In[237]:


train_predictions = model2.predict(X_train1).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
train_results


# In[238]:


import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][50:100])
plt.plot(train_results['Actuals'][50:100])


# Model 2:Predicted data using X_val1 vs data in y_val1.
# 

# In[239]:


val_predictions = model1.predict(X_val1).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val1})
val_results


# In[240]:


import matplotlib.pyplot as plt
plt.plot(val_results['Val Predictions'][0:100])
plt.plot(val_results['Actuals'][0:100])


# Model 2:Predicted data using X_test1 vs data in y_test1.
# 

# In[241]:


test_predictions = model2.predict(X_test1).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test1})
test_results


# In[242]:


import matplotlib.pyplot as plt
plt.plot(test_results['Test Predictions'][0:100])
plt.plot(test_results['Actuals'][0:100])


# In[243]:


#print("Mean Squared Error for Conv 1D model on test data:",mse(y_test1, test_predictions))


# Final thoughts:
# 
# Both of the models were able to predict the temperatures very well based upon the given data. Aside from the biological factors such as the sun and humidity there is correlation of the temperature rising due to the effect of the pollutants, heat from the engines etc. So we are sure that with the other factors like air quality we can predict the temperature at any given time of the day.
