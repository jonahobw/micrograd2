import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

import random
import numpy as np

from micrograd2.Value import Value
from micrograd2.nn import MLP

np.random.seed(1337)
random.seed(1337)

X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1 # make y be -1 or 1

# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')

# initialize a model 
model = MLP(2, 1, [16, 16])
params = model.parameters()
print(model)
print("number of parameters", len(model.parameters()))

# loss function
def loss(X, y, batch_size=10):
    total_loss = 0
    correct = 0
    preds = []

    # sample a batch from range [0, len(X)]
    batch_indices = np.random.choice(len(X), batch_size, replace=False)
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]

    for (xi, yi) in zip(X_batch, y_batch):
        score = model(xi)
        # svm "max-margin" loss
        loss = (1 + -yi*score).relu()
        # L2 regularization
        alpha = 1e-4
        reg_loss = alpha * sum((p*p for p in model.parameters()))
        total_loss += loss + reg_loss
        correct += int((yi > 0) == (score.val > 0))
        preds.append(score.val)
    
    return total_loss, correct, preds

total_loss, accuracy, _ = loss(X, y)
print(total_loss, accuracy)

# optimization
learning_rate = 0.005
batch_size = 10
for k in range(100):

    lr = learning_rate * 1/(1+0.01*k)
    model.reset_grad()
    total_loss, correct, _ = loss(X, y, batch_size)
    total_loss.backward()

    for p in model.parameters():
        p.val -= learning_rate * p.grad
    
    if k % 1 == 0:
        accuracy = 1.0 * correct / batch_size
        print(f"step {k} loss {total_loss.val:.4f}, {correct}/{batch_size} correct ({accuracy * 100}%)")
    
    if k % 10 == 0:
        total_loss, correct, _ = loss(X, y, batch_size=len(X))
        accuracy = 1.0 * correct / len(X)
        print(f"Total loss {total_loss.val:.4f}, Total {correct}/{batch_size} correct ({accuracy * 100}%)")
    


# visualize decision boundary
h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [list(map(Value, xrow)) for xrow in Xmesh]
scores = list(map(model, inputs))
Z = np.array([s.val > 0 for s in scores])
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()