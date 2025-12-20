import numpy as np


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
