# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:04:30 2020

@author: Basharat
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
seed = 7
np.random.seed(seed)

data = pd.read_csv("raw.csv")


print(data.head())

# ============================
from datetime import datetime
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

data = pd.read_csv('raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)

print(data.head())
# ============================

data = data.drop("No",axis=1)
data.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
data.index.name = 'date'
data['pollution'].fillna(0, inplace=True)

# drop the first 24 hours
data = data[24:]

data.to_csv("pollution.csv")

print(data.head())
# ============================

groups = [0, 1, 2, 3, 5, 6, 7]
values = data.values
fig,sub = plt.subplots(3,3)
plt.subplots_adjust(wspace=1, hspace=1)

for ax, i in zip(sub.flatten(),groups):
    ax.plot(values[:,i])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(data.columns[i])
    
# ============================
#preprocess the wind direction with label encoding
from sklearn.preprocessing import LabelEncoder
values = data.values
encoder = LabelEncoder()

values[:,4] = encoder.fit_transform(values[:,4])
values[:,4]

# ============================

#Scale the data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)
scaled[0]
# ============================

#convert data to supervised form

def to_supervised(data,dropNa = True,lag = 1):
    df = pd.DataFrame(data)
    column = []
    column.append(df)
    for i in range(1,lag+1):
        column.append(df.shift(-i))
    df = pd.concat(column,axis=1)
    df.dropna(inplace = True)
    features = data.shape[1]
    df = df.values
    supervised_data = df[:,:features*lag]
    supervised_data = np.column_stack( [supervised_data, df[:,features*lag]])
    return supervised_data

timeSteps = 3

supervised = to_supervised(scaled,lag=timeSteps)
pd.DataFrame(supervised).head()
# ============================

# spiltting the data
# training on only first year data
features = data.shape[1]
train_hours = 365*24
X = supervised[:,:features*timeSteps]
y = supervised[:,features*timeSteps]

x_train = X[:train_hours,:]
x_test = X[train_hours:,:]
y_train = y[:train_hours]
y_test = y[train_hours:]

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
# ============================

#convert data to fit for lstm
#dimensions = (sample, timeSteps here it is 1, features )

x_train = x_train.reshape(x_train.shape[0], timeSteps, features)
x_test = x_test.reshape(x_test.shape[0], timeSteps, features)

print( x_train.shape,x_test.shape)
# ============================

#define the LSTM model

model = Sequential()
model.add( LSTM( 50, input_shape = ( timeSteps,x_train.shape[2]) ) )
model.add( Dense(1) )

model.compile( loss = "mae", optimizer = "adam")

history =  model.fit( x_train,y_train, validation_data = (x_test,y_test), epochs = 50 , batch_size = 72, verbose = 1, shuffle = False)
# ============================

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.yticks([])
plt.xticks([])
plt.title("loss during training")
plt.show()

# ============================
#scale back the prediction to orginal scale
y_pred = model.predict(x_test)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[2]*x_test.shape[1])

inv_new = np.concatenate( (y_pred, x_test[:,-7:] ) , axis =1)
inv_new = scaler.inverse_transform(inv_new)
final_pred = inv_new[:,0]

y_test = y_test.reshape( len(y_test), 1)

inv_new = np.concatenate( (y_test, x_test[:,-7:] ) ,axis = 1)
inv_new = scaler.inverse_transform(inv_new)
actual_pred = inv_new[:,0]

# ============================

#plot the prediction with actual data

plt.plot(final_pred[:200], label = "prediction",c = "b")
plt.plot(actual_pred[:200],label = "actual data",c="r")
plt.xlim(0, 100)
plt.ylim(0, 300)
plt.yticks([])
plt.xticks([])
plt.title("comparison between prediction and actual data")
plt.legend()
# ============================

from sklearn.metrics import mean_absolute_error,mean_squared_error

print (("%f - mean absolute error")%(mean_absolute_error(final_pred,actual_pred) ))
print (("%f - mean squared error")%(mean_squared_error(final_pred,actual_pred)) )