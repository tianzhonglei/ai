# https://datascienceplus.com/keras-regression-based-neural-networks/

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Variables
#data in the formart : age,gender,miles,debt,income,sales
dataset = np.loadtxt("../data/cars.csv", delimiter=",")
x = dataset[:, 0:5]
y = dataset[:, 5]
y = np.reshape(y, (-1, 1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(x))
xscale = scaler_x.transform(x)
print(scaler_y.fit(y))
yscale = scaler_y.transform(y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

model = Sequential()
model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

history = model.fit(X_train, y_train, epochs=150, batch_size=50, verbose=1, validation_split=0.2)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

Xnew = np.array([[40, 0, 26, 9000, 8000]])

Xnew = np.array([[40, 0, 26, 9000, 8000]])
Xnew = scaler_x.transform(Xnew)
ynew = model.predict(Xnew)
# invert normalize
ynew = scaler_y.inverse_transform(ynew)
Xnew = scaler_x.inverse_transform(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
