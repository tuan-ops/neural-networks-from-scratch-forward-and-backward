"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
    x = np.asarray(x, dtype = float)
    numerical = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        tmp = x[idx]
        x[idx] = tmp + eps
        f_plus = f(x)
        x[idx] = tmp - eps
        f_minus = f(x)
        x[idx] = tmp
        numerical[idx] = ((f_plus - f_minus))/ (2 * eps)
    return numerical

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    analytic_grad = np.asarray(analytic_grad, dtype=float)
    numeric_grad = np.asarray(numeric_grad, dtype=float)

    numerator = np.abs(analytic_grad - numeric_grad)
    denominator = np.maximum(np.maximum(np.abs(analytic_grad), np.abs(numeric_grad)), tol)

    return float(np.max(numerator / denominator))

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # TODO: your approach here
    W, b = weight_init_fn(in_dim, out_dim)
    params = {"W": W, "b": b}
    def forward(x):
      y = x @ params["W"] + params["b"]
      cache = x
      return y, cache
    def backward(dout, cache):
      dx =  dout @ params["W"].T
      dW = cache.T @ dout 
      db = np.sum(dout, axis=0)
      grads = {"W": dW, "b": db}
      return dx, grads 
    return {"params": params, "forward": forward, "backward": backward}

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    # TODO: your approach here
    kind = kind.lower()
    if kind == "relu":
      def forward(x):
        y = np.maximum(x, 0)
        cache = x
        return y, cache 
      def backward(dout, cache):
        x = cache
        dx = dout * (x > 0)
        return dx, {}
    elif kind == "sigmoid":
      def forward(x):
        y = 1 / (1 + np.exp(-x))
        cache = x
        return y, cache 
      def backward(dout, cache):
        y = cache
        dx = dout * y * (1 - y)
        return dx, {}
    elif kind == "tanh":
      def forward(x):
        y = np.tanh(x)
        cache = x
        return y, cache 
      def backward(dout, cache):
        y = cache
        dx = dout * (1 - y ** 2)
        return dx, {}
    else:
        raise ValueError(f"Unsupported activation kind: {kind}")
    return {
        'params': {},
        'forward': forward,
        'backward': backward
    }

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    scheme = scheme.lower()
    if scheme in {"he", "he_normal", "kaiming", "kaiming_normal"}:
        std = np.sqrt(2 / in_dim)
        W = np.random.randn(in_dim, out_dim) * std 
    elif scheme in {"he_uniform", "kaiming_uniform"}:
        L = np.sqrt(6 / in_dim)   
        W = np.random.uniform(low  = -L, high = L, size = (in_dim, out_dim))
    elif scheme in {"lecun", "lecun_normal"}:
        std = np.sqrt(1 / in_dim)
        W = np.random.randn(in_dim, out_dim) * std
    elif scheme in {"lecun_uniform"}:
        L = np.sqrt(3 / in_dim)
        W = np.random.uniform(low = -L, high = L, size = (in_dim, out_dim))
    elif scheme in {"xavier", "xavier_normal", "glorot", "glorot_normal"}:
        std = np.sqrt(2 / (in_dim + out_dim))
        W = np.random.randn(in_dim, out_dim) * std 
    else:
        L = np.sqrt(6 / (in_dim + out_dim))
        W = np.random.uniform(low = - L, high = L, size = (in_dim, out_dim))
    b = np.zeros(out_dim, dtype = W.dtype)
    return W, b

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    # TODO: your approach here
    def loss_fn(logits, labels):
      N = logits.shape[0]
      logits = logits - np.max(logits, axis = 1, keepdims = True)
      probs = np.exp(logits) / np.sum(np.exp(logits), axis = 1, keepdims = True)
      correct_probs = probs[np.arange(N), labels]
      loss = -np.mean(np.log(correct_probs) + 1e-12)
      d_logits = probs.copy()
      d_logits[np.arange(N), labels] -= 1.0
      d_logits /= N 
      return loss, d_logits
    return loss_fn

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

