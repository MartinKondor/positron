import numpy as np

import activ
import score


"""
Settings the random seeds to the given value
"""
def seed(n):
    np.random.seed(n)


"""
Return activation functions and their derivatives 
from a list of strings.

:activfs_strings: activation functions as strings e.g. ["sigmoid", "relu"]
:return_derivate: bool
:returns: (activation_functions, deriv_activation_functions) if return_derivate else (activation_functions)
"""
def get_activactions_from_strings(activfs_strings, return_derivate=True):
    activfs = []
    dactivfs = []
    
    if isinstance(activfs_strings[0], str):
        for activfs_string in activfs_strings:
            try:
                activfs.append(getattr(activ, activfs_string))
                
                if return_derivate:
                    dactivfs.append(getattr(activ, "d" + activfs_string))
            
            except:
                raise Exception("Unknown activation function(s)")
    else:
        raise Exception("Unknown activation function(s)")

    if return_derivate:
        return activfs, dactivfs
    return activfs


"""
:cost_string: cost function's name as a string
:return_derivate: bool
:returns: (cost, dcost) if return_derivate else (cost)
"""
def get_cost_from_string(cost_string, return_derivate=True):
    cost = getattr(score, cost_string)
    if return_derivate:
        dcost = getattr(score, "d" + cost_string)
        return cost, dcost
    return cost


"""
:n_features: int, number of features
:weight_sizes: list with the size of the weights in order
:verbose: bool
:returns: (weights, biases) or ws, bs
"""
def init_network(n_features, weight_sizes, verbose=False):
    ws, bs = [], []
    w_row_number = n_features

    for s in weight_sizes:
        ws.append(np.random.random((w_row_number, s,)))
        bs.append(np.random.random((1, s,)))
        w_row_number = s

        if verbose:
            print("Weights created in shape:", ws[-1].shape)
            print("Biases created in shape:", bs[-1].shape)
            print()
    
    return ws, bs


"""
:a: input matrix
:ws: weights (list of np.ndarray)
:bs: biases (list of np.ndarray)
:actifs: list of activation functions
:return: the output of the network
"""
def feedforward(a, ws, bs, actifs_) -> np.ndarray:
    activfs = get_activactions_from_strings(actifs_, return_derivate=False)
    for w, b, actif in zip(ws, bs, activfs):
        a = actif(np.dot(a, w) + b)
    return a


"""
Inputs from update_batch function.
:rerurns: (nabla_ws, nabla_bs,)
"""
def backprop(x, y, ws, bs, activfs, dactivfs, dcost):
    nabla_ws = [np.zeros(w.shape) for w in ws]
    nabla_bs = [np.zeros(b.shape) for b in bs]
    layers = []
    a = x
    activated_layers = [x]

    # For each layer: (prev_activated_layer or X).w + b
    for w, b, activf in zip(ws, bs, activfs):
        z = np.dot(a, w) + b
        a = activf(z)

        layers.append(z)
        activated_layers.append(a)

    # For the first layer: dC/da * dA/dz
    delta_1 = dcost(a, y)
    delta_2 = dactivfs[-1](layers[-1])

    if len(delta_1.shape) == 1:
        delta_1 = delta_1.reshape((-1, 1))
    if len(delta_2.shape) == 1:
        delta_2 = delta_2.reshape((-1, 1))

    delta_layer = delta_1 * delta_2

    nabla_bs[-1] = delta_layer
    nabla_ws[-1] = np.dot(activated_layers[-2].T, delta_layer)
    
    # Then for other layers: (last_delta).w^(l-1).T * dA/dz^l
    for l in range(2, len(ws)):
        delta_layer = np.dot(delta_layer, ws[-l+1].T) * dactivfs[-l](layers[-l])
        nabla_bs[-l] = delta_layer
        nabla_ws[-l] = np.dot(activated_layers[-l-1].T, delta_layer)

    return nabla_ws, nabla_bs


"""
Update the weights on one mini_batch of data
Inputs from SGD function.

:x, y: [[...]_0 ... [...]_mini_batch_size]
:returns: ws, bs
"""
def update_batch(x, y, ws, bs, activfs, dactivfs, dcost, eta):
    nabla_ws = [np.zeros(w.shape) for w in ws]
    nabla_bs = [np.zeros(b.shape) for b in bs]
    
    # Backprop
    delta_nabla_ws, delta_nabla_bs = backprop(x, y, ws, bs, activfs, dactivfs, dcost)
    
    # Update nabla weights/biases
    nabla_ws = [nw+dnw for nw, dnw in zip(nabla_ws, delta_nabla_ws)]
    nabla_bs = [nb+dnb for nb, dnb in zip(nabla_bs, delta_nabla_bs)]

    # Update weights
    ws = [w - (eta/len(x) * nw) for w, nw in zip(ws, nabla_ws)]
    bs = [b - (eta/len(x) * nb) for b, nb in zip(bs, nabla_bs)]
    return ws, bs


"""
Stochastic Gradient Descent.

:X: input matrix
:y: the desired outputs
:ws: weights
:bs: biases
:activfs_: activation functions as a list of strings
:cost: cost function
:dcost: derivate of the cost function
:epoch: number of training sessions
:eta: learning rate
:returns: ws, bs, cost_history
"""
def SGD(X, y, ws, bs, activations, costf, epochs, eta, mini_batch_size=1, verbose=False, cost_history_needed=True):
    
    # First check the inputs
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(ws[0], np.ndarray):
        for i in range(len(ws)):
            ws[i] = np.array(ws[i])
    if not isinstance(bs[0], np.ndarray):
        for i in range(len(ws)):
            bs[i] = np.array(bs[i])

    if len(ws) == 0 or len(bs) == 0:
        raise Exception("No weights or biases given!")
    if len(ws) != len(bs):
        raise Exception("The number of weights and biases does not match!")
    if len(activations) == 0:
        raise Exception("No activation functions were given!")
    if len(activations) != len(ws):
        raise Exception("The number of activations and weights must be equal!")
    if not isinstance(activations[0], str):
        raise Exception("The activation functions must be given as strings!")
    if not isinstance(costf, str):
        raise Exception("The cost function must be given as a string!")

    # Start of SGD
    cost_history = []
    _batch_range = range(0, len(X), mini_batch_size)

    # Load activation/cost functions
    cost, dcost = get_cost_from_string(costf, return_derivate=True)
    activfs, dactivfs = get_activactions_from_strings(activations, return_derivate=True)

    # Run for each epoch
    _ep_range = range(epochs)
    if verbose:
        from tqdm import tqdm
        _ep_range = tqdm(_ep_range, desc=f"Training", unit=" ep")

    for epoch in _ep_range:

        # Choose baches
        mini_batches_x = X if mini_batch_size == len(X) else [X[k:k+mini_batch_size] for k in _batch_range]
        mini_batches_y = y if mini_batch_size == len(y) else [y[k:k+mini_batch_size] for k in _batch_range]
        
        # Train on batches 
        for mini_batch_x, mini_batch_y in zip(mini_batches_x, mini_batches_y):
            ws, bs = update_batch(mini_batch_x, mini_batch_y, ws, bs, activfs, dactivfs, dcost, eta)
        
        # Save cost history if needed
        if cost_history_needed:
            a = feedforward(X, ws, bs, activfs)
            cost_history.append(cost(a, y).sum())

    return ws, bs, cost_history


# Define an alias
evaluate = feedforward


if __name__ == "__main__":
    import example
    example.run()
