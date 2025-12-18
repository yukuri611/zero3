if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


lr = 0.2
iters = 5000
hidden_size = 10


model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y_pred, y)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if i % 100 == 0:
        print(f"iter: {i}, loss: {loss.data}")

x_test = np.linspace(0, 1, 100).reshape(100, 1)
y_test = model(x_test)
import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.plot(x_test, y_test.data, color="red")
plt.show()
