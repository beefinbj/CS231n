import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    self.params['b1'] = np.zeros((hidden_dim,))
    self.params['b2'] = np.zeros((num_classes,))
    self.params['W1'] = weight_scale * np.random.randn(input_dim,hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim,num_classes)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    affine1, cache1 = affine_forward(X, self.params['W1'], self.params['b1'])
    layer1, cache2 = relu_forward(affine1)
    affine2, cache3 = affine_forward(layer1, self.params['W2'], self.params['b2'])
    scores = affine2

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
      
    loss, dscores = softmax_loss(scores, y)
    W_reg = np.sum(self.params['W2']*self.params['W2']) + np.sum(self.params['W1']*self.params['W1'])
    loss += 0.5 * self.reg * W_reg
        
    drelu, dW2, db2 = affine_backward(dscores, cache3)
    grads['b2'] = db2
    grads['W2'] = dW2 + self.reg*self.params['W2']
    dout = relu_backward(drelu, cache2)
    dx, dW1, db1 = affine_backward(dout, cache1)
    grads['b1'] = db1
    grads['W1'] = dW1 + self.reg*self.params['W1']

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    
    for i in range(self.num_layers):
        Wname = 'W'+str(i)
        bname = 'b'+str(i)
        if i == 0:
            self.params[bname] = np.zeros((hidden_dims[i],))
            self.params[Wname] = weight_scale * np.random.randn(input_dim, hidden_dims[i])
        elif i == len(hidden_dims):
            self.params[bname] = np.zeros((num_classes,))
            self.params[Wname] = weight_scale * np.random.randn(hidden_dims[i-1], num_classes)
        else:
            self.params[bname] = np.zeros((hidden_dims[i],))
            self.params[Wname] = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])

    
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      for i in xrange(self.num_layers-1):
        self.bn_params.append({'mode': 'train'})
        self.params['gamma'+str(i)] = np.random.randn(hidden_dims[i],)
        self.params['beta'+str(i)] = np.zeros((hidden_dims[i]))
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    caches = {}
    
    layer_input = X
    
    for i in range(self.num_layers-1):
        W = self.params['W'+str(i)]
        b = self.params['b'+str(i)]
        FCout, cacheAffine = affine_forward(layer_input, W, b)
        caches['affine'+str(i)] = cacheAffine

        if self.use_batchnorm:
            gamma = self.params['gamma'+str(i)]
            beta = self.params['beta'+str(i)]
            FCout, cacheBatch = batchnorm_forward(FCout, gamma, beta, self.bn_params[i])
            caches['batch'+str(i)] = cacheBatch
               
        layerOut, cacheReLU = relu_forward(FCout)
        caches['ReLU'+str(i)] = cacheReLU
        
        if self.use_dropout:
            layerOut, cacheDrop = dropout_forward(layerOut, self.dropout_param)
            caches['drop'+str(i)] = cacheDrop
                   
        layer_input = layerOut
    
    W = self.params['W'+str(self.num_layers-1)]
    b = self.params['b'+str(self.num_layers-1)]
    affine, cacheAffine = affine_forward(layer_input, W, b)
    caches['affine'+str(self.num_layers-1)] = cacheAffine
           
    scores = affine

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    loss, dscores = softmax_loss(scores, y)
    
    reg_sum = 0    
    for i in range(self.num_layers):
        W = self.params['W'+str(i)]
        reg_sum += np.sum(W*W)
    loss += 0.5 * self.reg * reg_sum
    
    dout = dscores
    Wname = 'W'+str(self.num_layers-1)
    bname = 'b'+str(self.num_layers-1)
    cacheAffine = caches['affine'+str(self.num_layers-1)]
    drelu, dW, db = affine_backward(dout, cacheAffine)
    grads[Wname] = dW + self.reg*self.params[Wname]
    grads[bname] = db
    
    for i in range(self.num_layers-2,-1,-1):
        if self.use_dropout:
            drelu = dropout_backward(drelu, caches['drop'+str(i)])
        
        Wname = 'W'+str(i)
        bname = 'b'+str(i)
        Gname = 'gamma'+str(i)
        betaname = 'beta'+str(i)
        dFCout = relu_backward(drelu, caches['ReLU'+str(i)])
        
        if self.use_batchnorm:
            dFCout, dgamma, dbeta = batchnorm_backward_alt(dFCout, caches['batch'+str(i)])
            grads[Gname] = dgamma
            grads[betaname] = dbeta
            
        drelu, dW, db = affine_backward(dFCout, caches['affine'+str(i)])
        grads[Wname] = dW + self.reg*self.params[Wname]
        grads[bname] = db

    ############################################################################
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    return loss, grads
