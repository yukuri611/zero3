if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import matplotlib.pyplot as plt

import dezero.datasets
import dezero.functions as F
from dezero import DataLoader, optimizers
from dezero.models import MLP

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral(train=True)
test_set = dezero.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        acc = F.accuracy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    print("epoch: {}".format(epoch + 1))
    print(
        "train loss: {:.4f}, accuracy: {:.4f}".format(
            sum_loss / len(train_set), sum_acc / len(train_set)
        )
    )
    train_losses.append(sum_loss / len(train_set))
    train_accuracies.append(sum_acc / len(train_set))
    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy_simple(y, t)
            acc = F.accuracy(y, t)

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print(
        "test loss: {:.4f}, accuracy: {:.4f}".format(
            sum_loss / len(test_set), sum_acc / len(test_set)
        )
    )

    test_losses.append(sum_loss / len(test_set))
    test_accuracies.append(sum_acc / len(test_set))


plt.subplot(1, 2, 1)
plt.title("loss")
plt.plot(range(len(train_losses)), train_losses, label="train loss")
plt.plot(range(len(test_losses)), test_losses, label="test loss")
plt.subplot(1, 2, 2)
plt.title("accuracy")
plt.ylim(0, 1.0)
plt.plot(range(len(train_accuracies)), train_accuracies, label="train accuracy")
plt.plot(range(len(test_accuracies)), test_accuracies, label="test accuracy")
plt.show()
