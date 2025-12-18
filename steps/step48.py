if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math

import numpy as np

import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

x, t = dezero.datasets.get_spiral()
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

losses = []

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size : (i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print(f"epoch {epoch + 1}, loss {avg_loss:.4f}")
    losses.append(avg_loss)

import matplotlib.pyplot as plt

plt.plot(np.arange(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 1. 描画用のグリッド（網目）を作成
h = 0.1  # グリッドの間隔
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 2. グリッドの全点に対して予測を行う
X = np.c_[xx.ravel(), yy.ravel()]
with dezero.no_grad():  # 勾配計算をオフにする
    score = model(X)
predict_cls = np.argmax(score.data, axis=1)  # 一番確率が高いクラスを選択
Z = predict_cls.reshape(xx.shape)  # グリッドの形に戻す

# 3. グラフの描画
plt.figure(figsize=(8, 6))

# 背景をクラスごとに塗りつぶす
plt.contourf(xx, yy, Z)

# 元のデータポイントを散布図として重ねる
markers = ["o", "x", "^"]
colors = ["orange", "blue", "green"]
for i in range(3):  # クラス数 3
    mask = t == i
    plt.scatter(
        x[mask, 0], x[mask, 1], marker=markers[i], c=colors[i], label=f"class {i}"
    )

plt.title("Decision Boundary (MLP)")
plt.legend()
plt.show()
