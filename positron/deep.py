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
        bs.append(np.random.random((s, 1,)))
        
        if verbose:
            print("Created layer with shape:", ws[-1].shape, bs[-1].shape)
        
        prev_s = ws[-1].shape[1]

    return ws, bs


"""
:X: input matrix
:y: the desired outputs
:ws: weights
:bs: biases
:activf: activation functions
:dactifs: derivate of activation functions
:epoch: number of training sessions
:eta: learning rate
::
"""
def train(X, y, ws, bs, activf, dactivf, epoch, eta):
    cost_history = []
    a = np.copy(X)
    input_weights = []
    activated_weights = []

    for w, b in zip(ws[1:], bs[1:]):
        print(w.shape, b.shape, (w.shape[0], b.shape[1],))

        z = np.dot(a, ws[0]) + bs[0]
        a = activf[0](z)
        input_weights.append(z)
        activated_weights.append(a)

    # TODO

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
