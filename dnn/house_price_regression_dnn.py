#https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
#https://zhuanlan.zhihu.com/p/66182618

from pandas import read_csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# load dataset
dataframe = read_csv("../data/housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
X = dataset[:, 0:13]
Y = dataset[:, 13]
xtrain, xtest, ytrain, ytest=train_test_split(X, Y, test_size=0.15)

model = Sequential()
model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(xtrain, ytrain, batch_size=12, epochs=200, verbose=0)


ypred = model.predict(xtest)
#print(model.evaluate(xtrain, ytrain))
print(ytest)
print(ypred)

x_ax = range(len(ypred))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()
