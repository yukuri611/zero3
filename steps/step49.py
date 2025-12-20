if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import dezero
import dezero.datasets
import dezero.functions
from dezero import transforms

f = transforms.Normalize(mean=0.0, std=2.0)
train_set = dezero.datasets.Spiral(transform=f)
f = transforms.Compose(
    [transforms.Normalize(mean=0.0, std=2.0), transforms.AsType(np.float64)]
)
train_set = dezero.datasets.Spiral(transform=f)
