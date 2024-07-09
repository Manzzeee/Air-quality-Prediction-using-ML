# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:14:46 2021

@author: OKOK PROJECTS
"""

# For data reading | manipulation :
import pandas as pd 

# For reading the array :
import numpy as np

# For visualize the data and plottting patameters :
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15,8]
import seaborn as sns

# To suppress warnings :
from warnings import filterwarnings
filterwarnings('ignore')

# Import train-test split :
from sklearn.model_selection import train_test_split

# Import 'stats' libraries for modeling :
from scipy import stats
import statsmodels.api as sm
from statsmodels.api import OLS

# To scaled the data :
from sklearn.preprocessing import StandardScaler

# To check the accuracy of model :
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import Statsmodels :
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.arima_model import ARIMA

# Read the data :
AQI_city_hour = pd.read_csv('city_day.csv')


# Print the first five observations to see the data structure :
AQI_city_hour.head()

# Checking shape and data types of the data :

AQI_city_hour.info()

# Summary Statistics of the numeric data :

AQI_city_hour.describe()

from IPython import display 
display.Image("http://www.indiatogether.org/uploads/picture/image/2590/IT_airquality.png")

# Checking for the missing values and its precentage :

values = AQI_city_hour.isnull().sum()
percentage = 100*AQI_city_hour.isnull().sum()/len(AQI_city_hour)
table = pd.concat([values,percentage.round(2)],axis=1)
table.columns = ['No of missing values','Percent of missing values']
table[table['No of missing values']!=0].sort_values('Percent of missing values',ascending=False).style.background_gradient('OrRd')

# Check the missing values with heatmap :

sns.heatmap(AQI_city_hour.isnull(), cbar=False)
plt.show()

# Try methods to impute missing values :

AQI_city_hour.groupby(['City', 'Datetime'])['AQI'].mean()

# Converting Datetime variable into datetime data type :

AQI_city_hour['Datetime'] = AQI_city_hour['Datetime'].apply(pd.to_datetime)

# Check whether it is converted or not :

AQI_city_hour.dtypes

# Impute the missing values by grouping city with and average of 5 days:

data_fill = AQI_city_hour.iloc[:, :15].fillna(AQI_city_hour.iloc[:, :15].groupby(['City', pd.Grouper(key='Datetime', freq='5D')]).transform('mean'))

# After inputing missing values by day checking for the missing values :

values = data_fill.isnull().sum()
percentage = 100*data_fill.isnull().sum()/len(data_fill)
table = pd.concat([values,percentage.round(2)],axis=1)
table.columns = ['No of missing values','Percent of missing values']
table[table['No of missing values']!=0].sort_values('Percent of missing values',ascending=False).style.background_gradient('Purples')

# Impute the missing values by grouping city and month :

data_fill = data_fill.fillna(data_fill.groupby(['City', pd.Grouper(key='Datetime', freq='M')]).transform('mean'))

# After inputing missing values by month check for the missing values :

values = data_fill.isnull().sum()
percentage = 100*data_fill.isnull().sum()/len(data_fill)
table = pd.concat([values,percentage.round(2)],axis=1)
table.columns = ['No of missing values','Percent of missing values']
table[table['No of missing values']!=0].sort_values('Percent of missing values',ascending=False).style.background_gradient('Blues')

sns.heatmap(data_fill.isnull(), cbar=False)
plt.show()

# Imputing missing values by beckward fill :

data_fill = data_fill.fillna(method = 'bfill',axis=0)

# Still data is missing in the columns, hence using forward fill to impute this :

data_fill = data_fill.fillna(method = 'ffill',axis=0)

# After inputing missing values by bfill abd ffill:

values = data_fill.isnull().sum()
percentage = 100*data_fill.isnull().sum()/len(data_fill)
table = pd.concat([values,percentage.round(2)],axis=1)
table.columns = ['No of missing values','Percent of missing values']
table[table['No of missing values']!=0].sort_values('Percent of missing values',ascending=False).style.background_gradient('Blues')

# After imputing missing values, check summary statistics of the data :

data_fill.describe()

# Checking for outliers :

data_fill.plot(kind='box')
plt.title("Outliers in the Data")
plt.show()
plt.savefig('Outliers.png', dpi=300, bbox_inches='tight')

# In this we are not concidering the AQI_Bucket column, because it is not nessasary for our analysis.
# Here, we do analysis on all over country, so Aggregating the data into month wise and creating the new dataframe.

AQI_df = data_fill.groupby(['City', (data_fill.Datetime.dt.strftime('%Y-%m'))]).mean()

# Reset index :
AQI_df = AQI_df.reset_index()

# Check the First 5 observation :
AQI_df.head()

# Check the distribution of the all numerical columns and print skewness of the data :

AQI_df.drop(['City', 'Datetime'], axis=1).hist()
plt.tight_layout()
plt.title("Distribution of Data")
plt.show()
plt.savefig('Distribution of data.png')
# Print the skewness of the data :
print('Skewness :\n', AQI_df.drop(['City', 'Datetime'], axis=1).skew())

# Check the effect of all pollutants on AQI :

for i in AQI_df.iloc[:, 2:13]:
    print('The impact of ', i, 'on AQI')
    sns.scatterplot(x = i, y ='AQI', data = AQI_df,marker="o",sizes=200,color="r",label=i)
    plt.legend()
    plt.show()
    
    
# Correlation of the numerical data with heatmap :

sns.heatmap(AQI_df.corr(), annot=True, cmap='Blues')
plt.show()


# Yearly and Monthly Visualisation.

AQI_df.groupby(['Datetime']).mean().plot(kind='line', figsize=(15,20), subplots=True)

plt.show()

# Hourly Visualisations

data_fill.groupby(data_fill.Datetime.dt.strftime('%H')).mean().plot(kind='line', figsize=(15,15), subplots=True,linewidth=1.8)
plt.show()

# First seperate target and independent variables :
X = AQI_df.drop(['City', 'Datetime', 'AQI'], axis=1)
X = sm.add_constant(X)
y = AQI_df.AQI

# Fitting the model
model = OLS(y, X).fit()
model.summary()

# Print significant variables which are most affect on AQI :

significant = model.pvalues[model.pvalues < 0.05].sort_values(ascending=True)

print('Significant vaeiables which more affect on AQI :\n', significant)

# Import the library :
from statsmodels.tsa.stattools import adfuller

# Perform the test :
adfuller(AQI_df.AQI)

# Before this, Convert Datetime variable into Datetime data type in new dataframe AQI_df :

AQI_df['Datetime'] = AQI_df['Datetime'].apply(pd.to_datetime)

# Remove city from the dataframe as it is a Categorical column :
df = AQI_df.drop('City', axis=1)

# Set Datetime as index :
df.set_index('Datetime', inplace=True)

# Do groupby Datetime for average of AQI :
df1 = df.groupby('Datetime')['AQI'].mean()

# Decompose the time series :
decomposition = sm.tsa.seasonal_decompose(df1, model='additive')
fig = decomposition.plot()
plt.show()

# Import library ACF and PACF :
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

# PLot :
fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(df.AQI,lags=40,ax=ax1)

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df.AQI,lags=40,ax=ax2)
plt.show()

# Defining Function for Accuracy metrics
def forecast_accuracy(forecast, actual):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mape = np.mean(np.abs((actual - forecast)/actual))*100  # MAPE
    rmse = np.sqrt(mean_squared_error(actual,forecast))  # RMSE
    return({'mape':mape, 'rmse':rmse})

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

def VAR_accuracy(predicted, valid):
    from sklearn.metrics import mean_squared_error
    for i in df.columns:
        print("rmse value for ",i,"is :' ",np.sqrt(mean_squared_error(valid[i], predicted[i])))
        print("mape value for ",i,"is :' ", MAPE(valid[i].values,predicted[i]), end = '\n\n')
        
def VAR_AQI_MAPE(predicted, valid):
    return (MAPE(valid['AQI'].values,predicted['AQI']))

def VAR_AQI_RMSE(predicted, valid):
    return (np.sqrt(mean_squared_error(valid['AQI'], predicted['AQI'])))

# Prepare the data for Analysis :   (Purpose to make simple data for Time series)

month = data_fill.groupby([pd.Grouper(key='Datetime', freq='m')]).mean()

# Reset Index
month = month.reset_index()

# Setting Index
month = month.set_index('Datetime')

# Dropping the categorical
#week = week.drop('City', axis = 1)

# Check the dimensions of the data :
print(month.shape)

month.head()

#creating Train test split
train_month=month[:int(0.8*(len(month)))]
print('Train shape', train_month.shape)

valid_month=month[int(0.8*(len(month))):]
print('Test shape', valid_month.shape)

# Fitting the model

model = ARIMA(train_month.AQI,  order=(1,1,1)) 
model_fit=model.fit()
model_fit.summary()

# Import the library:
from pmdarima import auto_arima

stepwise_fit = auto_arima(train_month.AQI, trace = True, suppress_warnings = True, seasonal = False)

stepwise_fit.summary()

mod_month = sm.tsa.statespace.SARIMAX(train_month.AQI,
                                order=(3, 0, 2),
                                seasonal_order=(3, 0, 2, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=False)
result_month = mod_month.fit()
result_month.summary()

## Forecast for valid set and Future

pred_uc_month = result_month.get_forecast(steps=24)
pred_ci = pred_uc_month.conf_int()
ax = month.AQI.plot(label='observed', figsize=(14, 7))
pred_uc_month.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('AQI')
plt.title("Month wise prediction and forecast")
plt.legend()
plt.show()

# Calculate Accuracy Metrics

forecast_accuracy(pred_uc_month.predicted_mean[:15], valid_month.AQI)

from statsmodels.tsa.vector_ar.var_model import VAR

# Fitting the VAR model
VAR_model_month=VAR(endog=train_month)
VAR_model_month_fit=VAR_model_month.fit()

# Predicting on Validation set
predict_month=VAR_model_month_fit.forecast(VAR_model_month_fit.y,steps=len(valid_month))

pred_month=pd.DataFrame(predict_month,columns=df.columns,index=range(0,len(predict_month)))

VAR_accuracy(pred_month, valid_month)

VAR_AQI_MAPE(pred_month, valid_month)

VAR_AQI_RMSE(pred_month, valid_month)

# Prepare the data for Analysis :   (Purpose to make simple data for Time series)

week = data_fill.groupby([pd.Grouper(key='Datetime', freq='w')]).mean()

# Reset Index
week = week.reset_index()

# Setting Index
week = week.set_index('Datetime')

# Dropping the categorical
#week = week.drop('City', axis = 1)

# Check the dimensions of the data :
print(week.shape)

# Check first five values
week.head()

#creating Train test split
train_week=week[:int(0.8*(len(week)))]
print('Train shape', train_week.shape)

valid_week=week[int(0.8*(len(week))):]
print('Test shape', valid_week.shape)

from pmdarima import auto_arima

stepwise_fit_week = auto_arima(train_week.AQI, trace = True, suppress_warnings = True, seasonal = False)

stepwise_fit_week.summary()

mod_week = sm.tsa.statespace.SARIMAX(train_week.AQI,
                                order=(2, 0, 0),
                                seasonal_order=(2, 0, 0, 52),
                                enforce_stationarity=True,
                                enforce_invertibility=False)
result_week = mod_week.fit()
result_week.summary()

## Forecast for valid set and Future

pred_uc_week = result_week.get_forecast(steps=100)
pred_ci = pred_uc_week.conf_int()
ax = week.AQI.plot(label='observed', figsize=(14, 7))
pred_uc_week.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('AQI')
plt.legend()
plt.show()

