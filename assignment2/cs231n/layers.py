import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  N = x.shape[0]
  x_flat = x.reshape(N,-1)
  mult = x_flat.dot(w)
  out = mult+b.reshape(1,-1)
  
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  N = x.shape[0]
  x_flat = x.reshape(N,-1)
  dx = dout.dot(w.T)
  dx = dx.reshape(x.shape)
  dw = x_flat.T.dot(dout)
  db = dout.sum(axis=0)
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  out = x.clip(min=0)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  mask = x<=0
  dx = dout
  dx[mask] = 0
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    mini_mean = np.mean(x, axis=0)
    mini_var = np.var(x, axis=0)
    norm_x = (x-mini_mean)/(np.sqrt(mini_var+eps))
    out = norm_x*gamma+beta
    cache = (gamma, x, mini_mean, mini_var, eps, norm_x)
    running_mean = momentum * running_mean + (1 - momentum) * mini_mean
    running_var = momentum * running_var + (1 - momentum) * mini_var
    
  elif mode == 'test':
    out = (x-running_mean)/(np.sqrt(running_var+eps))
    out = out*gamma+beta
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  
  gamma, x, mini_mean, mini_var, eps, norm_x = cache
  dx = gamma*dout
  dx_cent = 1/(np.sqrt(mini_var+eps))*dx
  
  dstd = np.sum((x-mini_mean)*dx,axis=0)
  dstd = -1*((np.sqrt(mini_var+eps))**-2)*dstd
  
  dvar = 0.5*(mini_var)**(-0.5)*dstd
  dvar = dvar/(x.shape[0])
  dvar = 2*(x-mini_mean)*dvar
  dx_1 = dvar+dx_cent
  
  dmu = -1*np.sum(dx_1,axis=0)
  dmu = dmu/(x.shape[0])
  
  dx = dx_1 + dmu
    
  dgamma_mat = norm_x*dout
  dgamma = dgamma_mat.sum(axis=0)
  dbeta = dout.sum(axis=0)
  
  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  
  gamma, x, mini_mean, mini_var, eps, norm_x = cache
  
  dx_cent = 1/(np.sqrt(mini_var+eps))*gamma*dout
  dvar = -1*(mini_var+eps)**(-0.5)*(norm_x)*np.mean((norm_x)*gamma*dout,axis=0)
  
  dx = (dvar+dx_cent - np.mean(dx_cent,axis=0))
    
  dgamma = (norm_x*dout).sum(axis=0)
  dbeta = dout.sum(axis=0)
  
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    mask = np.random.rand(x.shape[1]) > p
    out = x*mask
    
  elif mode == 'test':
    out = x    

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    dx = dout*mask
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  N = x.shape[0]
  F = w.shape[0]
  H = x.shape[2]
  W = x.shape[3]
  HH = w.shape[2]
  WW = w.shape[3]
  pad = conv_param['pad']
  stride = conv_param['stride']
  H_out = 1+(H+2*pad-HH) / stride
  W_out = 1+(W+2*pad-WW) / stride
  X_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant', constant_values=0)
  out = np.zeros((N, F, H_out, W_out))
  for f in range(w.shape[0]):
      # C x HH x WW
      fil = w[f,:,:,:]
      for i in range(H_out):
          for j in range(W_out):
              # N x C x HH x WW
              chunk = X_pad[:,:,(stride*i):(stride*i+HH),(stride*j):(stride*j+WW)]
              conv = chunk*fil
              conv = np.sum(conv, axis=(1,2,3))
              conv = conv+b[f]
              out[:,f,i,j] = conv
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  
  x = cache[0]
  H = x.shape[2]
  W = x.shape[3]
  w = cache[1]
  HH = w.shape[2]
  WW = w.shape[3]
  pad = cache[3]['pad']
  stride = cache[3]['stride']
  
  db = np.sum(dout,axis=(0,2,3))
  
  x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant', constant_values=0)
  dx_pad = np.zeros_like(x_pad)
  dw = np.zeros_like(w)
  for f in range(w.shape[0]):
      fil = w[f,:,:,:]
      for i in range(dout.shape[2]):
          for j in range(dout.shape[3]):
              chunk = dout[:,f,i,j]
              rev_conv = fil*chunk[:,np.newaxis,np.newaxis,np.newaxis]
              dx_pad[:,:,(stride*i):(stride*i+HH),(stride*j):(stride*j+WW)] += rev_conv
              
              x_chunk = x_pad[:,:,(stride*i):(stride*i+HH),(stride*j):(stride*j+WW)]
              rev_x_pad = x_chunk*chunk[:,np.newaxis,np.newaxis,np.newaxis]
              dw_section = np.sum(rev_x_pad,axis=0)
              dw[f,:,:,:] += dw_section
  dx = dx_pad[:,:,pad:pad+H,pad:pad+W]
    
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  N, C, H, W = x.shape
  H_f = pool_param['pool_height']
  W_f = pool_param['pool_width']
  S = pool_param['stride']
  out = np.zeros((N, C, (H-H_f)/S+1, (W-W_f)/S+1))
  
  for i in range((H-H_f)/S+1):
      for j in range((W-W_f)/S+1):
          pool = x[:,:,(S*i):(S*i+H_f),(S*j):(S*j+W_f)]
          val = np.amax(pool,axis=(2,3))
          out[:,:,i,j] = val
  
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache[0]
  pool_param = cache[1]
  N, C, H, W = x.shape
  H_f = pool_param['pool_height']
  W_f = pool_param['pool_width']
  S = pool_param['stride']
  dx = np.zeros_like(x)
  
  for i in range((H-H_f)/S+1):
      for j in range((W-W_f)/S+1):
          pool = x[:,:,(S*i):(S*i+H_f),(S*j):(S*j+W_f)]
          pool_max = np.max(pool, axis=(2,3))
          binary_mask = (pool == (pool_max)[:,:,None,None])
          dx[:,:,(S*i):(S*i+H_f),(S*j):(S*j+W_f)] += binary_mask*(dout[:,:,i,j])[:,:,None,None]
  
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, C, H, W = x.shape
  running_mean = bn_param.get('running_mean', np.zeros((C,), dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros((C,), dtype=x.dtype))

  if mode == 'train':
    mini_mean = np.mean(x, axis=(0,2,3))
    mini_var = np.var(x, axis=(0,2,3))
    cent_x = x-mini_mean[np.newaxis,:,np.newaxis,np.newaxis]
    norm_x = cent_x/(np.sqrt(mini_var+eps)[np.newaxis,:,np.newaxis,np.newaxis])
    out = norm_x*gamma[np.newaxis,:,np.newaxis,np.newaxis]+beta[np.newaxis,:,np.newaxis,np.newaxis]
    cache = (gamma, x, mini_mean, mini_var, eps, norm_x)
    running_mean = momentum * running_mean + (1 - momentum) * mini_mean
    running_var = momentum * running_var + (1 - momentum) * mini_var
    
  elif mode == 'test':
    out = (x-running_mean[np.newaxis,:,np.newaxis,np.newaxis])/(np.sqrt(running_var+eps)[np.newaxis,:,np.newaxis,np.newaxis])
    out = out*gamma[np.newaxis,:,np.newaxis,np.newaxis]+beta[np.newaxis,:,np.newaxis,np.newaxis]
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None
  
  gamma, x, mini_mean, mini_var, eps, norm_x = cache
  
  dx_cent = 1/(np.sqrt(mini_var+eps)[np.newaxis,:,np.newaxis,np.newaxis])*gamma[np.newaxis,:,np.newaxis,np.newaxis]*dout
  dvar = -1*(mini_var+eps)[np.newaxis,:,np.newaxis,np.newaxis]**(-0.5)*(norm_x)*np.mean((norm_x)*gamma[np.newaxis,:,np.newaxis,np.newaxis]*dout,axis=(0,2,3))[np.newaxis,:,np.newaxis,np.newaxis]
  
  dx = (dvar+dx_cent - np.mean(dx_cent,axis=(0,2,3))[np.newaxis,:,np.newaxis,np.newaxis])
    
  dgamma = (norm_x*dout).sum(axis=(0,2,3))
  dbeta = dout.sum(axis=(0,2,3))
  
  
  return dx, dgamma, dbeta
  
  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
