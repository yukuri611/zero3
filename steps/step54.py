if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import dezero.functions as F
from dezero import test_mode

x = np.ones(5)
print(x)

# train mode
y = F.dropout(x)
print(y)

# test mode
with test_mode():
    y = F.dropout(x)
    print(y)
