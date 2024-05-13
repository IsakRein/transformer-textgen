import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.U = nn.Linear(input_size, hidden_size, dtype=float, bias=True)
        self.W = nn.Linear(hidden_size, hidden_size, dtype=float, bias=True)
        self.V = nn.Linear(hidden_size, output_size, dtype=float, bias=True)

        # Initialize weights using normal distribution scaled by 0.01
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.W.weight, std=0.01)
        nn.init.normal_(self.V.weight, std=0.01)

        # Initialize biases to zero (common practice)
        nn.init.zeros_(self.U.bias)
        nn.init.zeros_(self.W.bias)
        nn.init.zeros_(self.V.bias)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        output = torch.zeros((batch_size, self.input_size), dtype=torch.float)

        for i in range(batch_size):
            hidden = torch.tanh(self.U(input[i]) + self.W(hidden))
            output[i, :] = torch.log_softmax(self.V(hidden), dim=1)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, dtype=float)


def load_data(filepath):
    with open(filepath, 'r') as f:
        data = f.read()
    return np.array(list(data))


def synthesize(rnn, hprev, x0, n):
    h_t = hprev
    x_t = x0
    Y = torch.zeros((n, rnn.input_size), dtype=float)

    for t in range(n):
        output, h_t = rnn(x_t, h_t)
        p_t = output.detach().numpy()
        p_t = np.exp(p_t)
        p_t = p_t / np.sum(p_t)

        # generate x_t
        cp = np.cumsum(p_t)
        a = np.random.rand()
        ixs = np.where(cp - a > 0)[0]
        ii = ixs[0]

        # Update Y and x_t for next iteration
        Y[t, ii] = 1
        x_t = torch.zeros_like(x0)
        x_t[0, ii] = 1

    return Y


def train_rnn():
    torch.manual_seed(0)
    book_data = load_data('sample_data/goblet_book.txt')
    book_chars = np.unique(book_data)
    char_to_ind = {char: idx for idx, char in enumerate(book_chars)}
    ind_to_char = {idx: char for idx,
                   char in enumerate(book_chars)}

    K = len(book_chars)

    # TODO: Behöver en lägre lr än i min egen implementation.
    # Gissar att RMSprop är annorlunda på något sätt. Den lr för min egen implementation var 0.001.
    m = 100
    eta = 0.001
    gamma = 0.9
    seq_length = 25
    rnn = RNN(K, m, K)

    criterion = nn.CrossEntropyLoss(reduction='sum')

    # TODO: Undersök om detta är korrekt.
    optimizer = optim.RMSprop(rnn.parameters(), lr=eta)

    smooth_loss = None

    losses = []
    iterations = []

    iteration = 0
    epoch = 1

    temp = 0
    while epoch <= 3:
        print(f'-------------')
        print(f'Epoch {epoch}')

        rnn.zero_grad()
        hidden = rnn.initHidden()
        for i in range(0, len(book_data) - seq_length, seq_length):
            # Prepare inputs and targets
            X_chars = book_data[i:i+seq_length]
            Y_chars = book_data[i+1:i+seq_length+1]

            X = torch.zeros((seq_length, K), dtype=float)
            Y = torch.zeros((seq_length, K), dtype=float)

            # One-hot encode inputs and outputs
            for t, (x_char, y_char) in enumerate(zip(X_chars, Y_chars)):
                X[t, char_to_ind[x_char]] = 1
                Y[t, char_to_ind[y_char]] = 1

            # Forward pass
            output, hidden = rnn(X, hidden)

            # Backward pass
            loss = criterion(output, Y)

            # TODO: Undersök detta. Tror konsekvensen är att vi inte backpropagar genom hidden state.
            hidden = hidden.detach()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 0.001)
            optimizer.step()

            hidden

            if smooth_loss is None:
                smooth_loss = loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss

            if (iteration) % 100 == 0:
                losses.append(loss)
                iterations.append(iteration)

            if (iteration) % 1000 == 0:
                print(f'Iter {iteration:7d}. Smooth loss {
                      smooth_loss:7.2f}. Loss {loss:7.2f}.')

            if iteration % 10000 == 0:
                x0 = X[0, :].reshape(1, -1)
                hprev = rnn.initHidden()
                Y_synth = synthesize(rnn, hprev, x0, 200)

                txt = ''.join([ind_to_char[torch.argmax(y).item()]
                              for y in Y_synth])

                print(txt)

            iteration += 1
        epoch += 1

    return smooth_loss


if __name__ == '__main__':
    train_rnn()
