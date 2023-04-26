import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
import csv


sns.set()
plt.style.use('seaborn-whitegrid')

data = pd.read_csv(r'C:\Users\thatm\Downloads\mexicancurrency.csv')


#let us print the data head

print(data.head())

plt.figure(figsize=(10,4))
plt.title("MXN - USD Exchange Rate")
plt.xlabel("Date")
plt.ylabel("Close")
plt.plot(data["Close"])
plt.show()

#lets look at the correlations

print(data.corr()) #try to look up what these values insinuate
sns.heatmap(data.corr()) #visualization of what is printed

plt.show()

#prepare the dataset by storing the most relevant features in the variable x and storing the target column in the variable y 

#variable x = open, high, low

#variable y = close 

x = data[["Open", "High", "Low"]]
y=data["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1,1)

#now let's split the dataset and train a currency exchange prediction model using the value decision tree regression model using python 

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2, random_state = 42)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

data = pd.DataFrame(data={"Predicted Rate": ypred.flatten()})

data.head()