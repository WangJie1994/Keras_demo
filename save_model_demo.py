import numpy as np
np.random.seed(1111)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import load_model

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5*X + 2 + np.random.normal(0, 0.05, (200, ))

# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[: 160], Y[: 160]
X_test, Y_test = X[160: ], Y[160: ]

# build a neural network
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

# choose loss function
model.compile(loss='mse', optimizer='sgd')

# training
print('Training~~~')
for step in range(3001):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost:', cost)

# test
print('\nTesting~~~')
cost = model.evaluate(X_test, Y_test, 40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# ploting the prediciton
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.scatter(X_test, Y_pred)
plt.show()

print('test_before_save: ', model.predict(X_test[0:2]))
model.save('my_model.h5')
del model

# load
model = load_model('my_model.h5')
print('test_after_save: ', model.predict(X_test[0:2]))
