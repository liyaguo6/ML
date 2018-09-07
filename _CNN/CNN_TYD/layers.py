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
    # Reshape x into rows
    N = x.shape[0]
    x_row = x.reshape(N, -1)         # (N,D) 按照样本个数把数组拉直
    out = np.dot(x_row, w) + b       # (N,M)
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
    dx = np.dot(dout, w.T)                       # (N,D)
    # print(dx)
    dx = np.reshape(dx, x.shape)                 # (N,d1,...,d_k)
    # print(dx)
    x_row = x.reshape(x.shape[0], -1)            # (N,D)
    # print(x_row)
    dw = np.dot(x_row.T, dout)                   # (D,M)    
    db = np.sum(dout, axis=0, keepdims=True)     # (1,M)    

    return dx, dw, db

# x=np.random.randint(2,6,(3,10))
# print(x)
# w = np.random.randint(3,5,(10,5))
# b = np.zeros(5)
# dout=np.random.randint(2,3,(3,5))
# affine_backward(dout,(x,w,b))





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
    out = ReLU(x)    
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
    dx = dout    
    dx[x <= 0] = 0    

    return dx

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
    Computes the loss and gradient for softmax classification.    Inputs:    
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

def ReLU(x):
    """ReLU non-linearity."""    
    return np.maximum(0, x)
def conv_forward_naive(x, w, b, conv_param):
    """卷积层前向传播"""
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape  #N样本个数 #C当前输入channel #H 图像高  #W图像的高
    F, C, HH, WW = w.shape #F特征图的个数  #C特征图层数 #HH 特征图高 ##特征图的高
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    H_new = int(1 + (H + 2 * pad - HH) / stride)   #output大小
    W_new = int(1 + (W + 2 * pad - WW) / stride)   #output大小
    s = stride
    out = np.zeros((N, F, H_new, W_new))    #特征图

    for i in range(N):       # ith image
        for f in range(F):   # fth filter
            for j in range(H_new):
                for k in range(W_new):
                    #print x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s].shape
                    #print w[f].shape  
                    #print b.shape  
                    #print np.sum((x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] * w[f]))         
                    out[i, f, j, k] = np.sum(x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] * w[f]) + b[f]

    cache = (x, w, b, conv_param)

    return out, cache    #存储器


def  conv_backward_naive(dout, cache):
    """卷积反向传播"""
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_new = int(1 + (H + 2 * pad - HH) / stride)
    W_new = int(1 + (W + 2 * pad - WW) / stride)

    dx = np.zeros_like(x)

    dw = np.zeros_like(w)

    db = np.zeros_like(b)

    s = stride
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    for i in range(N):       # ith image
        for f in range(F):   # fth filter
            for j in range(H_new):
                for k in range(W_new):
                    window = x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s]
                    db[f] += dout[i, f, j, k]                
                    dw[f] += window * dout[i, f, j, k]                
                    dx_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] += w[f] * dout[i, f, j, k]

    # Unpad
    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]

    return dx, dw, db

# if __name__ == '__main__':
#     dout=np.random.randint(3,4,(2,4,32,32))
#     x=np.random.randint(5,7,(2,3,32,32))
#     w = np.random.randint(3,4,(4,3,7,7))
#     b=np.ones(4)
#     conv_param = {'stride': 1, 'pad': int((7 - 1) / 2)}
#     ret=conv_backward_naive(dout,(x,w,b,conv_param))
#     print(ret[0].shape)
#     print(ret[1].shape)
#     print(ret[2])



def max_pool_forward_naive(x, pool_param):
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = int(1 + (H - HH) / s)
    W_new = int(1 + (W - WW) / s)
    out = np.zeros((N, C, H_new, W_new))
    for i in range(N):
        for j in range(C):
            for k in range(H_new):
                for l in range(W_new):
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s] 
                    out[i, j, k, l] = np.max(window)

    cache = (x, pool_param)

    return out, cache


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = int(1 + (H - HH) / s)
    W_new = int(1 + (W - WW) / s)
    dx = np.zeros_like(x)
    for i in range(N):
        for j in range(C):
            for k in range(H_new):
                for l in range(W_new):
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s]                
                    m = np.max(window)               
                    dx[i, j, k*s:HH+k*s, l*s:WW+l*s] = (window == m) * dout[i, j, k, l]

    return dx