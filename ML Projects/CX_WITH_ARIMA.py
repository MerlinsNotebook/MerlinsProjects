import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\thatm\Downloads\mexicancurrency.csv')

data.head()

#let us focus on the date and close columns for this project 

data = data[["Date", "Close"]]

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,10))
plt.plot(data["Date"], data["Close"])

plt.show()


#determine if the data is seasonal or stationary
#from previous visualization we can see that the data does not look stationary

from statsmodels.tsa.seasonal import seasonal_decompose 

result = seasonal_decompose(data["Close"],
                             model = 'multiplicative', extrapolate_trend='freq',period=30)


fig = plt.figure()
fig = result.plot()
fig.set_size_inches(15,10)

plt.show()


#our data is seasonal meaning we have to use the SARIMA model 

pd.plotting.autocorrelation_plot(data["Close"])
plt.show()

#the correlation shows us that the curve is moving down after the 5th line!! 
# of the first boundary! This is how we decide the p-value 
#hence the value of p is 5 


#now lets find out the moving average 

from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(data["Close"], lags = 100)




