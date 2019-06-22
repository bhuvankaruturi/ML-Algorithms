import numpy as np

def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    return A, Z
    
def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

class NeuralNet:
    def __init__(self, L, layer_dims, non_linearities):
        self.L = L
        self.layer_dims = layer_dims
        self.non_linearities = non_linearities

        assert self.L == len(self.layer_dims) - 1 == len(self.non_linearities)

        print("Initializing a NN with %d layers"%(self.L))
        self.parameters = {}
        for l in range(1, self.L+1):
            self.parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            self.parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    def train(self, X, Y, learning_rate=0.01, iterations=1500, n_mini_batches=10):
        ''' Input:
            X - input feature matrix (n_x, number of examples)
            Y - vectors containing true labels (1, number of examples)
            learning_rate - learning rate of gradient descent, defaults to 0.01
            iterations - number of iterations of gradient descent updates, defaults to 1500
            n_mini_batches - number of mini batches to be made from the dataset, defaults to 10
            Return:
            None

            Method uses mini-batch gradient descent to fit a Neural Network to the given data
        '''
        m = X.shape[1]
        minibatches = range(0, m, int(m/int(n_mini_batches)))
        for i in range(iterations):
            for b in range(1, len(minibatches)):
                X_mini_batch = X[:, minibatches[b-1]:minibatches[b]]
                Y_mini_batch = Y[:, minibatches[b-1]:minibatches[b]]
                AL, caches = self.__forward_pass(X_mini_batch)
                if i%(iterations/10) == 0 and b == len(minibatches) - 1:
                    loss = self.compute_loss(AL, Y_mini_batch)
                    print("loss after %d iterations: %f"%(i, loss))
                grads = self.__backward_pass(AL, Y_mini_batch, caches)
                self.__update_parameters(grads, learning_rate)

    def predict(self, X):
        ''' Input:
            X - input feature matrix (n_x, number of examples) where n_x is number of features
            Output:
            predictions - (1, number of examples) vector with predictions of the network
        '''
        AL = self.__forward_pass(X)[0]
        return AL > 0.5
    
    def __forward_linear(self, A_prev, W, b):
        ''' Input:
            A_prev is a (n[l-1], number of examples) matrix of activations of previous layer
            W is a (n[l], n[l-1]) matrix of weights for the current layer
            b is a (n[l], 1) matrix of baises for the current layer 
            Output:
            Z is a (n[l], number of examples) matrix of intermediate inputs to be fed into a non-linearity
            cache is a tuple of (A_prev, W, b) to be used during backpropagation
        '''
        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)

        assert A_prev.dtype == W.dtype == b.dtype == 'float64' 

        return Z, cache
    
    def __forward_linear_activation(self, A_prev, W, b, non_linearity):
        ''' Input:
            A_prev is a (n[l-1], number of examples) matrix of activations of previous layer
            W is a (n[l], n[l-1]) matrix of weights for the current layer
            b is a (n[l], 1) matrix of baises for the current layer 
            Output:
            A is a (n[l], number of examples) matrix of activations of layer l
            cache is a tuple of ((A_prev, W, b), Z) to be used during backpropagation
        '''
        Z, linear_cache = self.__forward_linear(A_prev, W, b)
        if non_linearity == "sigmoid":
            A, activation_cache = sigmoid(Z)
        if non_linearity == "relu":
            A, activation_cache = relu(Z)
        cache = (linear_cache, activation_cache)

        assert A.dtype == 'float64'

        return A, cache

    def __forward_pass(self, X):
        ''' Input:
            X is a (n[l-1], number of examples) matrix of inputs
            Output:
            AL is a (n[L], number of examples) matrix of activations for last layer which are final class scores
            caches is list of caches of each layer computed during forward pass
        '''
        caches = []
        A = X

        for l in range(1, self.L + 1):
            A, cache = self.__forward_linear_activation(A, self.parameters["W" + str(l)], self.parameters["b" + str(l)], self.non_linearities[l-1])
            caches.append(cache)

        assert A.dtype == 'float64'

        return A, caches

    def compute_loss(self, AL, Y):
        ''' computes the cost/loss of the Neural Networks predictions'''
        m = AL.shape[1]
        #Avoid Divide by Zero error due to rouding of numbers
        AL[AL == 0] = 0.000001
        AL[AL == 1] = 0.999999

        assert AL.dtype == 'float64'
        assert np.sum(AL == 1) == 0
        assert np.sum(AL == 0) == 0

        loss = (np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)))/-m
        return loss

    def __backward_linear(self, dZ, cache):
        ''' Input:
            dZ is a (n[l], number of examples) matrix of gradients
            cache is a tuple of (A_prev, W, b) to be used in backpropagation
            Output:
            dA_prev is a (n[l-1], number of examples) matrix of gradients 
            dW is a (n[l], n[l-1]) matrix of gradients for the weight matrix of layer l
            db is a (n[l], 1) vector of gradients for the bias vector of layer l
        '''
        m = dZ.shape[1]
        A_prev, W, b = cache
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis = 1, keepdims = True) / m
        dA_prev = np.dot(W.T, dZ)

        assert dA_prev.dtype == dW.dtype == db.dtype == 'float64'

        return dA_prev, dW, db

    def __backward_linear_activation(self, dA, cache, non_linearity):
        ''' Input:
            dA is a (n[l], number of examples) matrix of gradients
            cache is a tuple of (A_prev, W, b) to be used in backpropagation
            non_linearlity is a string which indicates the non-linearity used in layer l
            Output:
            dA_prev is a (n[l-1], number of examples) matrix of gradients 
            dW is a (n[l], n[l-1]) matrix of gradients for the weight matrix of layer l
            db is a (n[l], 1) vector of gradients for the bias vector of layer l
        '''

        linear_cache, activation_cache = cache

        if non_linearity == "sigmoid":
            dZ = dA * (sigmoid(activation_cache)[0]) * (1 - sigmoid(activation_cache)[0])
        if non_linearity == "relu":
            dZ = dA * (activation_cache >= 0)
        dA_prev, dW, db = self.__backward_linear(dZ, linear_cache)

        assert dA_prev.dtype == dW.dtype == db.dtype == 'float64'

        return dA_prev, dW, db

    def __backward_pass(self, AL, Y, caches):
        ''' Input:
            AL is a (n[L], number of examples) matrix containing the output predictions of the NN
            Y is a (n[L], number of examples) matrix containing the true labels
            caches is a list of layer wise caches computed during forward pass
            Output:
            grads is a dictionary containing the gradients of loss function w.r.t weights and biases in the network
        '''

        grads = {}
        
        #Avoid Divide by Zero error due to rouding of numbers
        AL[AL == 0] = 0.000001
        AL[AL == 1] = 0.999999

        assert AL.dtype == 'float64'
        assert np.sum(AL == 1) == 0
        assert np.sum(AL == 0) == 0

        dAL = -np.divide(Y, AL) + np.divide((1 - Y), (1 - AL))

        dA_prev, grads["dW" + str(self.L)], grads["db" + str(self.L)] = self.__backward_linear_activation(dAL, caches[self.L-1], "sigmoid")

        for l in reversed(range(self.L-1)):
            dA_prev, grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = self.__backward_linear_activation(dA_prev, caches[l], "relu")

        return grads

    def __update_parameters(self, grads, learning_rate):
        ''' Input:
            grads - dictionary containing the gradients of NN loss function w.r.t the weigths and biases
            learning_rate - value with controls the magnitude of the update
            Returns:
            None

            This functions updates the parameters of the networks using the calculated gradients
        '''
        for l in range(1, self.L+1):
            self.parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]