import numpy as np


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
def init_network(input_shape: tuple or list, weight_sizes: list, output_size: int, verbose=False):
    ws, bs = [], []

    # Create weights
    prev_s = input_shape[-1]

    for s in weight_sizes + [output_size]:
        ws.append(np.random.random((prev_s, s,)))
        bs.append(np.random.random((1, s,)))
        
        if verbose:
            print("Created layer with shape:", ws[-1].shape, bs[-1].shape)
        
        prev_s = ws[-1].shape[1]

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
def SGD(X, y, ws, bs, activf, dactivf, cost, dcost, epoch, eta, verbose=False):
    cost_history = []

    for ep in range(epoch):
        a = np.copy(X)
        Z = []
        A = []

        # Feedforward
        for w, b, actf in zip(ws, bs, activf):
            z = np.dot(a, w) + b
            a = actf(z)
            Z.append(z)
            A.append(a)

        # Calculate cost
        c = cost(A[-1], y)
        cost_history.append(c)
        if verbose:
            print("Ep =", epoch, ", Cost =", c)

        # Calculate errors
        delta = dcost(A[-1], y) * dactivf[-1](Z[-1])
        nablas_w = [delta]
        nablas_b = [delta]

        for i, (z, w,) in enumerate(zip(Z[::-1][1:], ws[1:][::-1])):
            delta = np.dot(delta, w.T) * dactivf[-1](z)
            nablas_w.append(np.dot(delta.T, A[::-1][i]))
            nablas_b.append(delta)

        # Backprop
        # nabla[0] = nabla for last layer
        for w, b, nabla_w, nabla_b in zip(ws, bs, nablas_w[::-1], nablas_w[::-1]):
            print("ws,bs:", w.shape, b.shape, "nablas:", nabla_w.shape, nabla_b.shape)

    return ws, bs, cost_history


"""
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
def _SGD(X, y, ws, bs, activf, dactivf, cost, dcost, epoch, eta, verbose=False):
    cost_history = []

    for ep in range(epoch):
        a = np.copy(X)
        input_weights = []
        activated_weights = []

        # Feedforward
        for i, (w, b,) in enumerate(zip(ws, bs)):
            z = np.dot(a, w) + b
            a = activf[i](z)
            input_weights.append(z)
            activated_weights.append(a)

        # Calculate and save cost
        C = cost(a, y)
        cost_history.append(C)
        if verbose:
            print(f"epoch={ep}, C={C}")

        # First layer
        z = input_weights[-1]
        a = activated_weights[-1]
        delta_w = dcost(a, y) * dactivf[-1](a)
        deltas = [delta_w]

        # Skip the last layer from the loop
        # i = 0, 1, 2 ... len(activated_weights) - 2
        for i, a in enumerate(activated_weights[:-1]):
            delta_w = np.dot(delta_w, ws[-i-1].T) * dactivf[-i-1](a)
            deltas.append(delta_w)


        # Update the weights
        i = len(ws) - 1
        acs = [X, *activated_weights]
        for w in ws:

            # The first layer is updated with the input
            if i == 0:
                ws[0] += eta*np.dot(X.T, deltas[0])
                continue
            
            ws[-i] += eta*np.dot(activated_weights[i-1].T, deltas[i-1])
            i -= 1

    return ws, bs, cost_history


"""
:a: input matrix
:ws: weights (list of np.ndarray)
:bs: biases (list of np.ndarray)
:actifs: list of activation functions
:return: the output of the network
"""
def feedforward(a: np.ndarray, ws: list, bs: list, actifs: list):
    for i, (w, b) in enumerate(zip(ws, bs)):
        a = actifs[i](np.dot(a, w) + b)
    return a


if __name__ == "__main__":
    import example
    example.run()
