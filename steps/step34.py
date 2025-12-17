if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
from dezero import Variable

x = Variable(np.linspace(-7, 7, 100))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data.flatten()]

for i in range(3):
    logs.append(x.grad.data.flatten())
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

labels = ["y=sin(x)", "y'", "y''", "y'''", "y''''"]
for i, log in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc="upper left")
plt.show()
