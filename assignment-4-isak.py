import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy


class RNN:
    def __init__(self, K, m, eta, gamma, epsilon=1e-8):
        self.K = K
        self.m = m
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon

        self.p = {}  # Parameters: W, U, V, b, c
        self.mt = {}  # AdaGrad memory terms for each parameter

        # Initialize parameters with small random values
        self.p['b'] = np.zeros((m, 1))
        self.p['c'] = np.zeros((K, 1))
        self.p['U'] = np.random.randn(m, K) * 0.01
        self.p['W'] = np.random.randn(m, m) * 0.01
        self.p['V'] = np.random.randn(K, m) * 0.01

        # Initialize AdaGrad memory (mt) to zeros
        for key in self.p:
            self.mt[key] = np.zeros_like(self.p[key])

    def adagrad_update(self, grads):
        # Update each parameter according to AdaGrad rules
        for param in self.p.keys():
            # Accumulate the square of gradients
            self.mt[param] = (self.gamma * self.mt[param]) + \
                ((1-self.gamma) * grads.g[param] ** 2)
            # Perform the AdaGrad update for parameters
            self.p[param] -= (self.eta / np.sqrt(self.mt[param] +
                              self.epsilon)) * grads.g[param]


class RNNGradient:
    def __init__(self, rnn: RNN, n: int = 1):
        self.rnn = rnn
        self.n = n
        self.g = {}
        self.g['U'] = np.zeros_like(rnn.p['U'])
        self.g['W'] = np.zeros_like(rnn.p['W'])
        self.g['V'] = np.zeros_like(rnn.p['V'])
        self.g['b'] = np.zeros_like(rnn.p['b'])
        self.g['c'] = np.zeros_like(rnn.p['c'])


def load_data(filepath):
    with open(filepath, 'r') as f:
        data = f.read()
    return np.array(list(data))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def synthesize(rnn: RNN, h0: np.array, x0: np.array, n: int):
    h_t = h0
    xt = x0
    Y = np.zeros((rnn.K, n))

    assert x0.shape == (rnn.K, 1)
    assert h_t.shape == (rnn.m, 1)

    for t in range(n):
        # forward pass
        a_t = rnn.p['W'] @ h_t + rnn.p['U'] @ xt + rnn.p['b']
        assert a_t.shape == (rnn.m, 1)
        h_t = np.tanh(a_t)
        assert h_t.shape == (rnn.m, 1)
        o_t = rnn.p['V'] @ h_t + rnn.p['c']
        assert o_t.shape == (rnn.K, 1)
        p_t = softmax(o_t)
        assert p_t.shape == (rnn.K, 1)

        # generate xt
        cp = np.cumsum(p_t)
        a = np.random.rand()
        ixs = np.where(cp - a > 0)[0]
        ii = ixs[0]

        # Update Y and xt for next iteration
        Y[ii, t] = 1
        xt = np.zeros_like(x0)
        xt[ii] = 1

    return Y


def compute_loss(X, Y, RNN_try, hprev):
    P = forward_pass(RNN_try, hprev, X, Y)[2]
    return -np.sum(Y * np.log(P))


def forward_pass(rnn: RNN, h0: np.array, X: np.array, Y: np.array):
    n = X.shape[1]
    h_t = h0
    A = np.zeros((rnn.m, n))
    H = np.zeros((rnn.m, n))
    P = np.zeros((rnn.K, n))

    for t in range(n):
        # forward pass
        a_t = rnn.p['W'] @ h_t + \
            (rnn.p['U'] @ X[:, t]).reshape(-1, 1) + rnn.p['b']
        A[:, t] = a_t.reshape(-1)
        assert a_t.shape == (rnn.m, 1)

        h_t = np.tanh(a_t)
        H[:, t] = h_t.reshape(-1)
        assert h_t.shape == (rnn.m, 1)

        o_t = rnn.p['V'] @ h_t + rnn.p['c']
        assert o_t.shape == (rnn.K, 1)

        p_t = softmax(o_t)
        P[:, t] = p_t.reshape(-1)
        assert p_t.shape == (rnn.K, 1)

    return A, H, P


def compute_grads(rnn: RNN, X: np.array, Y: np.array, A: np.array, H: np.array, P: np.array):
    n = X.shape[1]
    grad = RNNGradient(rnn, n)

    dl_do = (P - Y)
    dl_dV = np.zeros_like(rnn.p['V'])
    for t in range(n):
        dl_dV += dl_do[:, t].reshape(-1, 1) @ H[:, t].reshape(1, -1)
    dl_dc = np.sum(dl_do, axis=1).reshape(-1, 1)

    dl_dh = np.zeros((rnn.m, n))
    dl_da = np.zeros((rnn.m, n))

    dl_dh[:, n-1] = dl_do[:, n-1] @ rnn.p['V']
    dl_da[:, n-1] = dl_dh[:, n-1] @ np.diag(1-(np.tanh(A[:, n-1])**2))

    for t in range(n-2, -1, -1):
        dl_dh[:, t] = dl_do[:, t] @ rnn.p['V'] + dl_da[:, t+1] @ rnn.p['W']
        dl_da[:, t] = dl_dh[:, t] @ np.diag(1-(np.tanh(A[:, t])**2))

    H_prev = np.hstack([np.zeros((rnn.m, 1)), H[:, :-1]])
    dl_dW = np.zeros_like(rnn.p['W'])
    dl_dU = np.zeros_like(rnn.p['U'])
    for t in range(n):
        dl_dW += dl_da[:, t].reshape(-1, 1) @ H_prev[:, t].reshape(1, -1)
        dl_dU += dl_da[:, t].reshape(-1, 1) @ X[:, t].reshape(1, -1)
    dl_db = np.sum(dl_da, axis=1).reshape(-1, 1)

    grad.g['U'] = dl_dU
    grad.g['V'] = dl_dV
    grad.g['W'] = dl_dW
    grad.g['b'] = dl_db
    grad.g['c'] = dl_dc

    for param in grad.g.keys():
        grad.g[param] = np.clip(grad.g[param], -5, 5)

    return grad


def compute_grads_num(X, Y, f, rnn: RNN, h):
    n = np.prod(rnn.p[f].shape)
    grad = np.zeros_like(rnn.p[f])
    hprev = np.zeros((rnn.m, 1))
    for i in range(n):
        rnn_try = copy.deepcopy(rnn)

        np.ravel(rnn_try.p[f])[i] -= h
        l1 = compute_loss(X, Y, rnn_try, hprev)

        np.ravel(rnn_try.p[f])[i] += 2*h
        l2 = compute_loss(X, Y, rnn_try, hprev)

        grad.ravel()[i] = (l2 - l1) / (2 * h)
    return grad


def compare_gradients():
    np.random.seed(0)
    book_data = load_data('goblet_book.txt')
    book_chars = np.unique(book_data)

    char_to_ind = {char: idx for idx, char in enumerate(book_chars)}
    ind_to_char = {idx: char for idx, char in enumerate(book_chars)}

    K = len(book_chars)
    m = 100
    eta = 0.1
    gamma = 1
    seq_length = 3

    rnn = RNN(K, m, eta, gamma)

    X_chars = book_data[0:seq_length]
    Y_chars = book_data[1:seq_length+1]

    X = np.zeros((K, seq_length))
    Y = np.zeros((K, seq_length))

    for i in range(seq_length):
        X[char_to_ind[X_chars[i]], i] = 1
        Y[char_to_ind[Y_chars[i]], i] = 1

    h0 = np.zeros((rnn.m, 1))

    A, H, P = forward_pass(rnn, h0, X, Y)
    grads = compute_grads(rnn, X, Y, A, H, P)

    grads_num = {}
    h = 1e-4
    for param in ['U', 'W', 'V', 'b', 'c']:
        grads_num[param] = compute_grads_num(X, Y, param, rnn, h)

    for param in grads.g.keys():
        abs_error = np.abs(grads.g[param] - grads_num[param])
        rel_error = abs_error / \
            (np.maximum(
                1e-6, np.abs(grads.g[param]) + np.abs(grads_num[param])))

        print(f'Gradient comparison for {param}:')
        print(f'Absolute error: {np.mean(abs_error)}')
        print(f'Relative error: {np.mean(rel_error)}\n')


def train_rnn():
    np.random.seed(0)
    book_data = load_data('data/goblet_book.txt')
    book_chars = np.unique(book_data)

    char_to_ind = {char: idx for idx, char in enumerate(book_chars)}
    ind_to_char = {idx: char for idx, char in enumerate(book_chars)}

    K = len(book_chars)
    m = 100
    eta = 0.001
    gamma = 0.9
    seq_length = 25

    rnn = RNN(K, m, eta, gamma)

    smooth_loss = None
    lowest_loss = None

    losses = []
    iterations = []

    iteration = 0
    epoch = 1

    temp = 0

    while epoch <= 3:
        print(f'-------------')
        print(f'Epoch {epoch}')

        hprev = np.zeros((rnn.m, 1))
        for i in range(0, len(book_data) - seq_length, seq_length):
            # Prepare inputs and targets
            X_chars = book_data[i:i+seq_length]
            Y_chars = book_data[i+1:i+seq_length+1]

            X = np.zeros((rnn.K, seq_length), dtype=int)
            Y = np.zeros((rnn.K, seq_length), dtype=int)

            # One-hot encode inputs and outputs
            for t, (x_char, y_char) in enumerate(zip(X_chars, Y_chars)):
                X[char_to_ind[x_char], t] = 1
                Y[char_to_ind[y_char], t] = 1

            # Forward pass
            A, H, P = forward_pass(rnn, hprev, X, Y)
            
            # Backward pass
            grads = compute_grads(rnn, X, Y, A, H, P)

            # AdaGrad parameter update
            rnn.adagrad_update(grads)

            # Compute loss and smooth loss
            loss = compute_loss(X, Y, rnn, hprev)
            if smooth_loss is None:
                smooth_loss = loss
                lowest_loss = loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss

            # Update the hidden state (hprev) to the last state of this pass
            hprev = H[:, -1].reshape(-1, 1)

            if (iteration) % 100 == 0:
                losses.append(loss)
                iterations.append(iteration)

            if (iteration) % 1000 == 0:
                print('iter %d, smooth_loss %f' %
                      (iteration, smooth_loss))

            if iteration % 10000 == 0:
                print('iter %d, smooth_loss %f, loss %f' %
                      (iteration, smooth_loss, loss))
                x0 = X[:, 0].reshape(-1, 1)
                Y_synth = synthesize(rnn, hprev, x0, 200)
                txt = ''.join([ind_to_char[ind]
                               for ind in np.argmax(Y_synth, axis=0)])
                print(txt)

            iteration += 1
        epoch += 1

    plt.plot(iterations, losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iteration')
    plt.show()

    return smooth_loss


# compare_gradients()
train_rnn()
