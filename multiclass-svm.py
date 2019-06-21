import numpy as np

def LossFunction(X, y, W):
    #X is in the form N * M, where N is number of examples and M is the number of features + 1
    #y is in the form N * 1
    #W is in the form K * M
    delta = 1.0
    reg_param = 0.5
    scores = np.dot(X, W.T) # N * K
    wrong_class = np.ones(scores.shape)
    wrong_class[np.arange(wrong_class.shape[0]), y.reshape(y.size)] = 0
    margins = (scores - scores[np.arange(scores.shape[0]).reshape(scores.shape[0],1), y] + delta) * wrong_class
    loss = np.sum(np.sum(np.maximum(0, margins)))/margins.shape[0] + reg_param * (np.sum(np.sum(W[:,1:]**2)))
    return loss

X = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [3, 5, 7, 8], [10, 12, 11, 13]])
y = np.array([1, 2, 0, 2]).reshape(4, 1)
W = np.absolute(np.random.randn(3, 4))

print(LossFunction(X, y, W))