import numpy as np
from tqdm import tqdm


"""
Settings the random seeds to the given value
"""
def seed(n: int):
    np.random.seed(n)


"""
:input_shape:
:weight_sizes: size of the weights in order
:output_size: the number of outputs
"""
def init_network(input_shape: tuple or list, weight_sizes: list, verbose=False):
    ws, bs = [], []

    w_row_number = input_shape[1]
    for s in weight_sizes:
        w = np.random.random((w_row_number, s,))
        b = np.random.random((1, s,))
        w_row_number = s
        ws.append(w)
        bs.append(b)

        if verbose:
            print("Weights created in shape:", w.shape)
            print("Biases created in shape:", b.shape)
            print()
    
    return ws, bs


"""
:a: input matrix
:ws: weights (list of np.ndarray)
:bs: biases (list of np.ndarray)
:actifs: list of activation functions
:return: the output of the network
"""
def feedforward(a: np.ndarray, ws: np.ndarray, bs: np.ndarray, actifs: list) -> np.ndarray:
    for w, b, actif in zip(ws, bs, actifs):
        a = actif(np.dot(a, w) + b)
    return a


"""
Inputs from update_batch function.
:rerurns: (nabla_ws, nabla_bs,)
"""
def backprop(x, y, ws, bs, activfs, dactivfs, cost, dcost, eta):
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
    delta_layer = dcost(a, y) * dactivfs[-1](layers[-1])
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
def update_batch(x, y, ws, bs, activfs, dactivfs, cost, dcost, eta):
    nabla_ws = [np.zeros(w.shape) for w in ws]
    nabla_bs = [np.zeros(b.shape) for b in bs]
    
    # Backprop
    delta_nabla_ws, delta_nabla_bs = backprop(x, y, ws, bs, activfs, dactivfs, cost, dcost, eta)
    
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
:activf: activation functions
:dactifs: derivate of activation functions
:cost: cost function
:dcost: derivate of the cost function
:epoch: number of training sessions
:eta: learning rate
"""
def SGD(X, y, ws, bs, activfs, dactivfs, cost, dcost, epochs, eta, mini_batch_size=1, verbose=False, cost_history_needed=True):
    cost_history = []
    _batch_range = range(0, len(X), mini_batch_size)

    # Run for each epoch
    _ep_range = range(epochs)
    if verbose:
        _ep_range = tqdm(_ep_range, desc=f"Epoch[{0}]", unit=" ep")

    for epoch in _ep_range:

        # Choose baches
        mini_batches_x = [X[k:k+mini_batch_size] for k in _batch_range]
        mini_batches_y = [y[k:k+mini_batch_size] for k in _batch_range]
        
        for mini_batch_x, mini_batch_y in zip(mini_batches_x, mini_batches_y):
            ws, bs = update_batch(mini_batch_x, mini_batch_y, ws, bs, activfs, dactivfs, cost, dcost, eta)
            
        if cost_history_needed:
            a = feedforward(X, ws, bs, activfs)
            cost_history.append(cost(a, y).sum())

    return ws, bs, cost_history


if __name__ == "__main__":
    import example
    example.run()
