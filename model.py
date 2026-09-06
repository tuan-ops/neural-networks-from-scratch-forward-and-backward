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

# Step 3 - make_dense (not yet solved)
# TODO: implement

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

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

