
import pandas as pd
import numpy as np
import seaborn as sns
import math
import tensorflow
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers import LSTM
from keras.layers import Dense,Dropout,RepeatVector,Activation
from sklearn.preprocessing import StandardScaler



dt = pd.read_csv('AirQualityUCI.csv',delimiter=';',decimal = ',')


# What do the instances that comprise the dataset represent?
# 
# 0 Date	(DD/MM/YYYY)
# 
# 1 Time	(HH.MM.SS)
# 
# 2 True hourly averaged concentration CO in mg/m^3  (reference analyzer)
# 
# 3 PT08.S1 (tin oxide)  hourly averaged sensor response (nominally  CO targeted)	
# 
# 4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
# 
# 5 True hourly averaged Benzene concentration  in microg/m^3 (reference analyzer)
# 
# 6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)	
# 
# 7 True hourly averaged NOx concentration  in ppb (reference analyzer)
# 
# 8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted) 
# 
# 9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)	
# 
# 10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)	
# 
# 11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
# 
# 12 Temperature in Â°C	
# 
# 13 Relative Humidity (%) 	
# 
# 14 AH Absolute Humidity


dt = dt.drop(columns= ['Unnamed: 15','Unnamed: 16'])



dt.isna().sum()





dt= dt.dropna()
dt



dt.describe()




dt['Time'].str.replace('.',':',regex= False)



dt



dt['Date']=dt['Date'].str.replace('/','-')




train_dates = dt['Date']




train_dates = pd.to_datetime(dt['Date']+ " "+ dt['Time'].str.replace('.',':',regex=False))



dt['Date'] = train_dates




dt['Time']= dt['Time'].str.replace('.00.00','',regex = False).astype(float)



dt["t_norm"] = 2 * math.pi * dt["Time"] / dt["Time"].max()
dt['Sin Hour'] = np.sin(dt['t_norm'])
dt['Cos Hour'] = np.cos(dt['t_norm'])




dt= dt.sort_values(by = 'Date')




dt = dt.drop(columns=['t_norm'])




graph = dt.loc[(dt['Date']>'2004-01-04') & (dt['Date']<'2004-01-10')]
sns.lineplot(x = graph['Date'],y = graph['CO(GT)'])



dt.head(10)


cols_data = list(dt)[2:]
cols_data



dt_train = dt[cols_data].astype(float)



dt_train.describe()




cols_scaled = list(dt)[2:15]+ list(dt)[-6:]
scaler = StandardScaler()
scaler = scaler.fit(dt_train[cols_scaled])
dt_train_scaled= scaler.transform(dt_train[cols_scaled])




dt_train_scaled.shape



Xtrain = []
Ytrain = []
timesteps = 6
future_days = 1


# **Create a "sliding window" of values using past days as an interval for the 13 columns**



for i in range(timesteps, len(dt_train_scaled)- future_days+1):
    Xtrain.append(dt_train_scaled[i-timesteps:i,0:dt_train_scaled.shape[1]])
    Ytrain.append(dt_train_scaled[i+future_days-1:i+future_days,0])
    



trainX, trainY = np.array(Xtrain),np.array(Ytrain)


# **You can see how Ytrain is taking the next interval of 6 after every 12 days**



print('Shape of train X {}',trainX.shape)
print('Shape of train Y {}',trainY.shape)


# **Creating the Model**



model = Sequential()
model.add(LSTM(64,activation = 'relu',input_shape = (trainX.shape[1],trainX.shape[2]), return_sequences= True))
model.add(LSTM(32,activation = 'relu', return_sequences= False))
model.add(Dropout(.5))
model.add(Dense(trainY.shape[1]))
model.compile(optimizer = 'adam',loss= 'mse')




results = model.fit(trainX, trainY, epochs = 25, batch_size = 64,validation_split=.1,verbose = 1)



model.summary()




plt.plot(results.history['val_loss'])
plt.plot(results.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['val_loss', 'loss'], loc='upper left')
plt.show()


# **Forecasting the future**



future_hours = 200
future_dates = pd.date_range(list(train_dates)[-1],periods = future_hours, freq = '1H').tolist()
forecast= model.predict(trainX[-future_hours:])




forecast_copies = np.repeat(forecast,len(cols_scaled),axis=-1)
ypred = scaler.inverse_transform(forecast_copies)[:,0]



fdates= [] 
for t in future_dates:
    fdates.append(t)
finaldt= pd.DataFrame({'Date':np.array(fdates),'CO(GT)':ypred})
finaldt['Date']=pd.to_datetime(finaldt['Date'])



graph_dt =dt.loc[(dt['Date']>'2005-04-04 02:00:00') & (dt['Date']<'2005-04-04 15:00:00')]
sns.lineplot(x = graph_dt['Date'],y = graph_dt['CO(GT)'])
sns.lineplot(x = finaldt['Date'],y = finaldt['CO(GT)'])

