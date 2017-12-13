import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  for i in xrange(num_train):
      f = X[i].dot(W)
      correct_f = f[y[i]]
      log_term = 0
      stability = -max(f)
      for j in xrange(num_classes):
          term = np.exp(f[j]+stability)
          log_term += term
          dW[:,j] += term/np.sum(np.exp(f+stability))*X[i].T
          if j == y[i]:
              dW[:,j] -= X[i].T
      loss += np.log(log_term)
      loss -= correct_f+stability

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg*W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  f = X.dot(W)
  max_f = np.amax(f,axis=1)
  shifted_f = f-max_f[:,np.newaxis]
  correct_f = shifted_f[np.arange(num_train),y]
  loss_i = np.zeros(num_train)
  loss_i -= correct_f
  loss_i += np.log(np.sum(np.exp(shifted_f),axis=1))
  
  loss = sum(loss_i)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  coeffs_i = np.ones((num_train,num_classes))
  coeffs_i *= np.exp(shifted_f)
  coeffs_i /= np.sum(np.exp(shifted_f),axis=1)[:,np.newaxis]
  coeffs_i[np.arange(num_train),y] -= 1 
  
  dW = X.T.dot(coeffs_i)
  dW /= num_train
  dW += reg*W 

  return loss, dW

