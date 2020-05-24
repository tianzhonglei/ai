
#https://www.datatechnotes.com/2019/12/how-to-fit-regression-data-with-cnn.html

from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#load data
dataframe = read_csv("../data/housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
X = dataset[:, 0:13]
Y = dataset[:, 13]
#stand data
sc = StandardScaler()
X = sc.fit_transform(X)
#reshape data
X = X.reshape(X.shape[0], X.shape[1], 1)
#split data
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.15)

model = Sequential()
model.add(Conv1D(128, 2, activation="relu", input_shape=(13,1)))
model.add(Conv1D(128, 2, activation="relu"))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
model.summary()
model.fit(xtrain, ytrain, batch_size=12, epochs=200, verbose=0)

ypred = model.predict(xtest)
print(model.evaluate(xtrain, ytrain))
print("MSE: %.4f" % mean_squared_error(ytest, ypred))

x_ax = range(len(ypred))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()