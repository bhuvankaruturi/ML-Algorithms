import numpy as np

def gradient(X, y, W):
    n = X.shape[0]
    gradients = (np.dot(X.T, (np.dot(X, W) - y)))/(n)
    return gradients

def costFunction(X, y, W):
    n = X.shape[0] # number of instances   
    return (np.sum((np.dot(X, W) - y) ** 2))/(2*n)

def gradientDescent(X, y, W, alpha):
    itr = 1
    gradients = gradient(X, y, W)
    print("Cost after iteration", itr, "is", costFunction(X, y, W))
    while itr < 1000:
        W = W - alpha * gradients
        gradients = gradient(X, y, W)
        itr = itr + 1
    print("Cost after iteration", itr, "is", costFunction(X, y, W))
    return W

X = np.random.randint(50, size=(200, 3))
bias = np.random.randint(5, size=(200, 1))
y = np.array(2 * X[:,0] - 4 * X[:,1] + 3 * X[:,2] + 15)
y = y.reshape((y.shape[0], 1))
X = np.hstack((np.ones((X.shape[0], 1)) * 100, X))
W = np.random.rand(X.shape[1], 1)
print("initial random weights")
print(W)
print("------")

W = gradientDescent(X, y, W, 0.0001)
print(W)
