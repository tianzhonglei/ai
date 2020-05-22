from keras import models
from keras import layers

import numpy as np


model = models.Sequential([layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer="sgd", loss="mean_squared_error")

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
xy = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, xy, epochs=500)

print(model.predict([10.0]))

'''
# simple predict
import numpy as np
from keras.layers import Input, Flatten
from keras.models import Model
inputs = Input(shape=(3,2,4))

# Define a model consisting only of the Flatten operation
prediction = Flatten()(inputs)
model = Model(inputs=inputs, outputs=prediction)

X = np.arange(0,24).reshape(1,3,2,4)
print(X)
#[[[[ 0  1  2  3]
#   [ 4  5  6  7]]
#
#  [[ 8  9 10 11]
#   [12 13 14 15]]
#
#  [[16 17 18 19]
#   [20 21 22 23]]]]
model.predict(X)
#array([[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
#         11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,
#         22.,  23.]], dtype=float32)
'''
