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

#softmax linear classifier
#intialization
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))

reg = 1e-3 #regularization strength
step_size = 1e-0

for i in range(200):

    #computing class scores
    scores = np.dot(X, W) + b

    #computing the cross-entropy loss for softmax classifier
    num_examples = X.shape[0]
    exp_scores = np.exp(scores)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

    correct_logprobs = -np.log(probs[range(num_examples), y])

    data_loss = np.sum(correct_logprobs)/num_examples
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss

    if i % 10 == 0:
        print("iteration %d: loss %f"%(i, loss))

    #computing the analytic gradient
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    #using chain rule to back-propogate the gradient
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg*W

    #performing a parameter update
    W += -step_size * dW
    b += -step_size * db

#evaluating accuracy after training
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print("training accuracy: %.2f"%(np.mean(predicted_class == y)))