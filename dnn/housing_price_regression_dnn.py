#https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
#https://zhuanlan.zhihu.com/p/66182618

from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# load dataset
dataframe = read_csv("../data/housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:, 0:13]
Y = dataset[:, 13]

x_test = np.array(X[int(len(X) * 0.96):])

model = Sequential()
model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=100,batch_size=5, verbose=0)

print('--------x_test------------')
print(x_test)
print('------predict_value-------')
print(model.predict(x_test))
