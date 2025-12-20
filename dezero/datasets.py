import gzip

import matplotlib.pyplot as plt
import numpy as np

from dezero.transforms import Compose, Flatten, Normalize, ToFloat
from dezero.utils import get_file


class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.data = None
        self.label = None
        self.transform = transform
        if self.transform is None:
            self.transform = lambda x: x
        self.target_transform = target_transform
        if self.target_transform is None:
            self.target_transform = lambda x: x
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)  # index is int
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(
                self.label[index]
            )

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(train=self.train)


def get_spiral(seed=1984, train=True):
    np.random.seed(seed)
    N = 100  # クラスごとのサンプル数
    DIM = 2  # データの要素数
    CLS_NUM = 3  # クラス数

    x = np.zeros((N * CLS_NUM, DIM))
    t = np.zeros((N * CLS_NUM), dtype=int)

    for j in range(CLS_NUM):
        for i in range(N):
            rate = i / N
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            t[ix] = j

    return x, t


class MNIST(Dataset):
    def __init__(
        self,
        train=True,
        transform=Compose([Flatten(), ToFloat(), Normalize(0.0, 255.0)]),
        target_transform=None,
    ):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        # url = 'http://yann.lecun.com/exdb/mnist/'
        url = "https://ossci-datasets.s3.amazonaws.com/mnist/"  # mirror site
        train_files = {
            "target": "train-images-idx3-ubyte.gz",
            "label": "train-labels-idx1-ubyte.gz",
        }
        test_files = {
            "target": "t10k-images-idx3-ubyte.gz",
            "label": "t10k-labels-idx1-ubyte.gz",
        }

        files = train_files if self.train else test_files
        data_path = get_file(url + files["target"])
        label_path = get_file(url + files["label"])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, "rb") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H : (r + 1) * H, c * W : (c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)
                ].reshape(H, W)
        plt.imshow(img, cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.show()

    @staticmethod
    def labels():
        return {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
        }
