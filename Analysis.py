# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 20:33:21 2021

@author: OKOK PROJECTS
"""

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import folium

from geopy.geocoders import Nominatim



import io
city_df=pd.read_csv("D:/PYTHON/STUDENT/AIRAnalysis/city_day123.csv")

# Display the top 5 rows of the dataframe
city_df.head()

# Get the dimensions of the dataframe
print("The number of rows and columns in the dataframe = ",city_df.shape)

missing_df = (city_df.isnull().sum()).to_frame()          # Calculate the number of missing values in each column
missing_df.reset_index(inplace=True)                      # Reset index
missing_df.columns=['col_name','missing_val']             # Rename the columns
missing_df.sort_values(by = 'missing_val', inplace=True)  # Sort in ascending order
missing_df 

# Visualizing the frequency of missing values
colors = cm.rainbow(np.linspace(0, 1, len(missing_df['col_name']))) # Get a range of colors
plt.figure(figsize=(10,5)) # Set the figure size
plt.bar(missing_df['col_name'],missing_df['missing_val'], color=colors) # Plot the bar graph
plt.title('Number of missing values in each column', size=17) # Set the title
# Configuring the axes
plt.xticks(fontsize=12,rotation=90)
plt.yticks(fontsize=12)
plt.xlabel('Parameters',color='green',size=15)
plt.ylabel('Frequency of Missing Values',color='red',size=15)
plt.show()

# Get the basic information about each column in the city_df dataframe
city_df.info()

# Missing values
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing values in descending order
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("The dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

missing_values= missing_values_table(city_df)
missing_values.style.background_gradient(cmap='Blues')  #,axis = 0)



cities = city_df['City'].value_counts()   # Count the number of observations for each city
print(f'Total number of cities in the dataset : {len(cities)}')  # Count the total number of cities in the dataframe
print(cities)

# Visualize the number of observations for each city
plt.figure(figsize=(15,5))    # Set the figure size
plt.plot(cities, 'r--') # Plot a line graph for the number of observations in each city
plt.title("Number of Observation in each City", size=17)
# Configure the axes
plt.xticks(fontsize=12,rotation=90)
plt.yticks(fontsize=12)
plt.xlabel('Name of the City',color='green',size=15)
plt.ylabel('Number of Observation',color='red',size=15)
plt.grid(True, c='green')

# Convert string to datetime64
city_df['Date'] = pd.to_datetime(city_df['Date'])

# Range of availability of data
print(f"The available data is between {city_df['Date'].min()} and {city_df['Date'].max()}")
# Create a new column combining few existing ones and drop the older ones
city_df['BTX'] = city_df['Benzene']+city_df['Toluene']+city_df['Xylene']
city_df.drop(['Benzene','Toluene','Xylene'],axis=1, inplace=True)

# Create a new column combining few existing ones
city_df['Particulate_Matter'] = city_df['PM2.5']+city_df['PM10']

city_df.columns  # A glimpse at the columns in the dataframe

pollutants = ['NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2','O3', 'BTX','Particulate_Matter']   # Make a list of pollutants

boxplot = city_df.boxplot(column=[pollutant for pollutant in pollutants], figsize=(15,8))  # Create a boxplot to understand data distribution

def trend_plot(dataframe,value):
    
    # Prepare data
    df['year'] = [d.year for d in df.Date]
    df['month'] = [d.strftime('%b') for d in df.Date]
    years = df['year'].unique()

    # Draw Plot
    fig, axes = plt.subplots(1, 2, figsize=(14,3), dpi= 150)
    sns.boxplot(x='year', y=value, data=df, ax=axes[0])
    sns.pointplot(x='month', y=value, data=df.loc[~df.year.isin([2015, 2020]), :], color='#F08080')

    # Configure the title and the axes
    axes[0].set_xlabel('Year', fontsize=14, color = '#ff8000') #'#aa00ff')
    axes[1].set_xlabel('Month', fontsize = 14, color = '#ff8000')  #aa00ff')  
    axes[0].set_ylabel(value, color = '#ff0066')
    axes[1].set_ylabel(value, color = '#ff0066')
    axes[0].set_title('Year-wise Box Plot \n(The Trend) ', fontsize=14, color='#8600b3')   #'#00bfff')
    axes[1].set_title('Month-wise Plot \n(The Seasonality)', fontsize=14, color='#8600b3')  #'#00bfff')
    plt.show()
    
city_df.reset_index(inplace=True)
df = city_df.copy()

for i in pollutants:
  trend_plot(df,i)
  
plt.figure(figsize=(12,7))
sns.heatmap(city_df.isnull(), cmap="YlGnBu")

def max_polluted_city(pollutant):
    x1 = city_df[[pollutant,'City']].groupby(["City"]).mean().sort_values(by=pollutant,ascending=False).reset_index()
    x1[pollutant] = round(x1[pollutant],2)
    print(x1[:10].style.background_gradient(cmap='OrRd'))
    return x1[:10].style.background_gradient(cmap='OrRd')

from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.render()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    
pm = max_polluted_city('Particulate_Matter')
no2 = max_polluted_city('NO2')
so2 = max_polluted_city('SO2')
co = max_polluted_city('CO')
btx = max_polluted_city('BTX')


display_side_by_side(pm,no2,so2,co,btx)

cities = ['Thiruvananthapuram','Najafgarh','Indirapuram','VivekVihar','SoniaVihar','Rohini']   # List of major Indian cities
filtered_city_df = city_df[city_df['Date'] >= '2019-01-01']  # Filter data after 1.1.2019
AQI = filtered_city_df[filtered_city_df.City.isin(cities)][['Date','City','AQI','AQI_Bucket']]
AQI.head()

AQI_pivot = AQI.pivot(index='Date', columns='City', values='AQI')
AQI_pivot.fillna(method='bfill',inplace=True)


fig = make_subplots(
    rows=6, cols=1,
    #specs=[[{}, {}],
          # [{"colspan": 6}, None]],
    subplot_titles=("Thiruvananthapuram","Indirapuram","Rohini","Najafgarh",'SoniaVihar','VivekVihar'))

fig.add_trace(go.Bar(x=AQI_pivot.index, y=AQI_pivot['Thiruvananthapuram'],
                    marker=dict(color=AQI_pivot['Thiruvananthapuram'],coloraxis="coloraxis")),
              1, 1)
fig.add_trace(go.Bar(x=AQI_pivot.index, y=AQI_pivot['Indirapuram'],
                    marker=dict(color=AQI_pivot['Indirapuram'], coloraxis="coloraxis")),
              2, 1)
fig.add_trace(go.Bar(x=AQI_pivot.index, y=AQI_pivot['Rohini'],
                    marker=dict(color=AQI_pivot['Rohini'], coloraxis="coloraxis")),
              3, 1)
fig.add_trace(go.Bar(x=AQI_pivot.index, y=AQI_pivot['Najafgarh'],
                    marker=dict(color=AQI_pivot['Najafgarh'], coloraxis="coloraxis")),
              4, 1)
fig.add_trace(go.Bar(x=AQI_pivot.index, y=AQI_pivot['SoniaVihar'],
                    marker=dict(color=AQI_pivot['SoniaVihar'], coloraxis="coloraxis")),
              5, 1)
fig.add_trace(go.Bar(x=AQI_pivot.index, y=AQI_pivot['VivekVihar'],
                    marker=dict(color=AQI_pivot['VivekVihar'], coloraxis="coloraxis")),
              6, 1)

fig.update_layout(coloraxis=dict(colorscale='Temps'),showlegend=False,title_text="AQI Levels")

fig.update_layout(plot_bgcolor='white')

fig.update_layout( width=800,height=1200,shapes=[
      dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2020-03-25', x1= '2020-03-25'
    )
])

fig.show()
AQI_beforeLockdown = AQI_pivot['2020-01-01':'2020-03-25']  # AQI levels before lockdown
AQI_afterLockdown = AQI_pivot['2020-03-26':'2020-05-01']  # AQI levels after lockdown

print(AQI_beforeLockdown.mean())  # Mean AQI for cities before lockdown
print(AQI_afterLockdown.mean())   # Mean AQI for cities after lockdown

filtered_city_day = city_df[city_df['Date'] >= '2019-01-01']
AQI = filtered_city_day[filtered_city_day.City.isin(cities)][['Date','City','AQI','AQI_Bucket']]

start_date1 = '2019-01-01'
end_date1 = '2019-05-01'

mask1 = (city_df['Date'] >= start_date1) & (city_df['Date']  <= end_date1)
pollutants_filtered_2019 = city_df.loc[mask1]
pollutants_filtered_2019.fillna(method='bfill',inplace=True)
pollutants_filtered_2019.set_index('Date',inplace=True);

start_date2 = '2020-01-01'
end_date2 = '2020-05-01'

mask2 = (city_df['Date'] >= start_date2) & (city_df['Date']  <= end_date2)
pollutants_filtered_2020 = city_df.loc[mask2]
pollutants_filtered_2020.fillna(method='bfill',inplace=True)
pollutants_filtered_2020.set_index('Date',inplace=True);


df1 = pollutants_filtered_2019[pollutants_filtered_2019.City.isin(cities)][['City','NO2','SO2','PM2.5','CO']]
df2 = pollutants_filtered_2020[pollutants_filtered_2020.City.isin(cities)][['City','NO2','SO2','PM2.5','CO']]

df1 = pollutants_filtered_2019[pollutants_filtered_2019.City.isin(cities)][['City','NO2','SO2','PM2.5','CO']]
df2 = pollutants_filtered_2020[pollutants_filtered_2020.City.isin(cities)][['City','NO2','SO2','PM2.5','CO']]



def pollution_comparison(city):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df1.index, y=df1[df1['City']==city]['NO2'],
                    line=dict(dash='solid',color='green'),name='NO2'))
    fig.add_trace(go.Scatter(x=df1.index, y=df1[df1['City']==city]['SO2'],
                    line=dict(dash='dot',color='red'),name='SO2'))
    fig.add_trace(go.Scatter(x=df1.index, y=df1[df1['City']==city]['PM2.5'],
                    line=dict(dash='dashdot',color='dodgerblue'),name='Particulate_Matter'))
    fig.add_trace(go.Scatter(x=df1.index, y=df1[df1['City']==city]['CO'],
                    line=dict(dash='longdashdot'),mode='lines',name='CO'))
    fig.update_layout(title_text=city+' 2019 ',plot_bgcolor='white')
    fig.update_xaxes(rangeslider_visible=True,zeroline=True,zerolinewidth=1, zerolinecolor='Black')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df2.index, y=df2[df2['City']==city]['NO2'],
                    line=dict(dash='solid',color='green'),name='NO2'))
    fig.add_trace(go.Scatter(x=df2.index, y=df2[df2['City']==city]['SO2'],
                    line=dict(dash='dot',color='red'),name='SO2'))
    fig.add_trace(go.Scatter(x=df2.index, y=df2[df2['City']==city]['PM2.5'],
                    line=dict(dash='dashdot',color='dodgerblue'),name='Particulate_Matter'))
    fig.add_trace(go.Scatter(x=df2.index, y=df2[df2['City']==city]['CO'],
                    line=dict(dash='longdashdot'),mode='lines',name='CO'))
    fig.update_layout(title_text=city+' 2020 ',plot_bgcolor='white')
    fig.update_xaxes(rangeslider_visible=True,zeroline=True,zerolinewidth=1, zerolinecolor='Black')
    fig.show()
    
    
pollution_comparison('Thiruvananthapuram')

city_df.head()

city = city_df.dropna()   # Drop any NaN values

X = city[['O3','PM2.5','PM10','CO','SO2','NO2']]  # Independent Columns
y = city['AQI']   # Dependent Column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)  # Employs train-test split

regr = linear_model.LinearRegression()
model = regr.fit(X_train, y_train)  # Fit the model
print(regr.coef_)

pred = regr.predict(X_test)  # Predict AQI from the model

print('r2_score:',r2_score(y_test,pred))

from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 
for i in range(2,5):
    poly = PolynomialFeatures(degree = i) 
    X_poly = poly.fit_transform(X_train) 

    poly.fit(X_poly, y_train) 
    lin2 = LinearRegression() 
    lin2.fit(X_poly, y_train) 
    res = lin2.predict(poly.fit_transform(X_test))
    print('degree = ',i,'r2_score:',r2_score(y_test,res))
    
poly = PolynomialFeatures(degree = 3) 
X_poly = poly.fit_transform(X_train) 

poly.fit(X_poly, y_train) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y_train) 
res = lin2.predict(poly.fit_transform(X_test))
print('degree = ',i,'r2_score:',r2_score(y_test,res))

# Visualize the number of observations for each city
plt.figure(figsize=(15,5))    # Set the figure size
# Scatter plot for original values
plt.scatter(X_test.index,y_test, alpha=0.5, c='#58D68D', label='original')
# Scatter plot for predicted values
plt.scatter(X_test.index,pred, alpha=0.15, c='#3498DB', label='prediction')
plt.title('Actual AQI vs. Predicted AQI for Multiple Linear Regression', size=17)
# Configure the axes
plt.xticks(fontsize=12,rotation=90)
plt.yticks(fontsize=12)
plt.xlabel('Index',color='green',size=15)
plt.ylabel('AQI',color='green',size=15)
plt.legend(loc='upper right', borderaxespad=0.5)
plt.show()

# Visualize the number of observations for each city
plt.figure(figsize=(15,5))    # Set the figure size
# Scatter plot for original values
plt.scatter(X_test.index,y_test, c='red', label='original')
# Scatter plot for predicted values
plt.scatter(X_test.index, res , alpha=0.35, c='yellow', label='prediction')
plt.title('Actual AQI vs. Predicted AQI for Polynomial Regression', size=17)
# Configure the axes
plt.xticks(fontsize=12,rotation=90)
plt.yticks(fontsize=12)
plt.xlabel('Index',color='red',size=15)
plt.ylabel('AQI',color='red',size=15)
plt.legend(loc='upper right', borderaxespad=0.5)
plt.show()