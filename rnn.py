import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from gensim.models import Word2Vec
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

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
        output = torch.zeros((batch_size, self.output_size), dtype=torch.float)
        for i in range(batch_size):
            hidden = torch.tanh(self.U(input[i]) + self.W(hidden))
            output[i, :] = torch.log_softmax(self.V(hidden), dim=1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, dtype=float)

def load_word2vec(use_fewer_dimensions=False):
    if use_fewer_dimensions:
        model_path = "word_embeddings/word2vec_80.model"
        word2vec_path = "word_embeddings/word_vectors_80.npy"
    else:
        model_path = "word_embeddings/word2vec.model"
        word2vec_path = "word_embeddings/word_vectors.npy"
    # The gensim model.
    model = Word2Vec.load(model_path)
    
    # K x N numpy array where K is the number of features for a word and N is the number of words in the corpus.
    word2vec = np.load(word2vec_path)

    # List of all words that appear in chronological order.
    words = np.load("word_embeddings/words.npy")

    # Set of all unique words in the corpus. Is actually list(set) so it supports indexing.
    word_set = np.load("word_embeddings/word_set.npy")

    # Keypairs of words in word_set and their index in word_set
    with open("word_embeddings/word_to_index.pkl", "rb") as f:
        word_to_index = pickle.load(f)

    return model, word2vec, words, word_set, word_to_index

def load_data(filepath):
    with open(filepath, 'r') as f:
         data = f.read()
    return np.array(list(data))

def synthesize_word2vec(rnn, hprev, x0, n, word_set, model):
    h_t = hprev
    x_t = x0
    Y = [""] * n
    
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
        word_vec = model.wv[word_set[ii]]

        # Update Y and x_t for next iteration
        Y[t] = word_set[ii]

        x_t = torch.reshape(torch.tensor(word_vec), x_t.shape)
        x_t = x_t.double()
    return Y

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

def test_word2vec_seq(model, word2vec_data, words):
    for i in range(len(words)):
        assert np.array_equal(model.wv[words[i]], word2vec_data[:, i])

def construct_Y_batch(start_index, end_index, words, word_set, seq_length, word_to_index):
    Y = torch.zeros((seq_length, word_set.shape[0]))
    i = 0
    for j in range(start_index, end_index):
        index = word_to_index[words[j]]
        Y[i, index] = 1
        i += 1
    return Y

def train_rnn_word2vec():
    torch.manual_seed(0)
    model, word2vec, words, word_set, word_to_index = load_word2vec(use_fewer_dimensions=False)
    test_word2vec_seq(model=model, word2vec_data=word2vec, words=words)
    
    # Dimension of word vector
    K = word2vec.shape[0]
    num_words = word2vec.shape[1]
    output_size = word_set.shape[0]

    # TODO: Behöver en lägre lr än i min egen implementation.
    # Gissar att RMSprop är annorlunda på något sätt. Den lr för min egen implementation var 0.001.
    m = 100
    eta = 0.001
    gamma = 0.9
    seq_length = 25
    
    rnn = RNN(K, m, output_size)

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
        for i in range(0, num_words - seq_length, seq_length):
            # Prepare inputs and targets
            X = word2vec[:, i:i+25].T
            X = torch.tensor(X)
            Y = construct_Y_batch(start_index=i+1, end_index=i+26, words=words, word_set=word_set, seq_length=seq_length, word_to_index=word_to_index)

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
                print(f'Iter {iteration:7d}. Smooth loss {smooth_loss:7.2f}. Loss {loss:7.2f}.')

            if iteration % 1000 == 0:
                x0 = X[0, :].reshape(1, -1)
                hprev = rnn.initHidden()
                Y_synth = synthesize_word2vec(rnn, hprev, x0, 20, word_set, model)
                txt = " ".join(Y_synth)
                print(txt)

            iteration += 1
        epoch += 1

    return smooth_loss


def train_rnn():
    torch.manual_seed(0)
    book_data = load_data('word_embeddings/goblet_book.txt')
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

    criterion = nn.CrossEntropyLoss(reduction='mean')

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
                print(f'Iter {iteration:7d}. Smooth loss {smooth_loss:7.2f}. Loss {loss:7.2f}.')

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
    #train_rnn()
    train_rnn_word2vec()
