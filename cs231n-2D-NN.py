''' Implementation of a 2 layer Neural Network '''

import numpy as np
import matplotlib.pyplot as plt

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# visualize the data:
#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()

#creating a 2 layered NN (i.e one hidden layer)
h = 100 #size of the hidden layer
W1 = 0.01 * np.random.randn(D, h)
b1 = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

reg = 1e-3
step_size = 1e-0

for i in range(10000):
    #evaluating class scores 
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
    scores = np.dot(hidden_layer, W2) + b2

    #computing the cross-entropy loss
    num_examples = X.shape[0]
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    correct_logprobs = -np.log(probs[range(num_examples), y])

    data_loss = np.sum(correct_logprobs)/num_examples
    reg_loss = 0.5*reg*(np.sum(W1*W1) + np.sum(W2*W2))
    loss = data_loss + reg_loss

    if i % 1000 == 0:
        print("iteration: %d loss: %f"%(i, loss))

    #computing the gradient of scores
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    #back-propagating the gradient into second layer's parameters
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)

    dhidden = np.dot(dscores, W2.T)
    #gradient of ReLU d/dx(max(0, x))= 1(x>0) or 0(x<=0)
    dhidden[hidden_layer<=0] = 0

    #backpropagating the gradient to first layer's parameters
    dW1 = np.dot(X.T, dhidden)
    db1 = np.sum(dhidden, axis=0, keepdims=True)

    #adding the regularization loss gradient
    dW2 += reg*W2
    dW1 += reg*W1

    #performing the parameter updates
    W1 += -step_size*dW1
    b1 += -step_size*db1
    W2 += -step_size*dW2
    b2 += -step_size*db2

#training accuracy of the learning parameters
hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print("training accuracy: %.2f"%(np.mean(predicted_class == y)))

#ploting the resulting classifier
interval = 0.002
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, interval), (y_min, y_max, interval))

Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W1) + b1 ), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha = 0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
fig.savefig('spiral_net.png')