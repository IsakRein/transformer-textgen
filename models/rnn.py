import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import pickle
import sys
import json
from gensim.models import Word2Vec


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


def load_data(tokenizer):
    data = torch.tensor(
        np.load(f"token_data/text_{tokenizer}.npy"), dtype=torch.long)
    with open(f"token_data/vocabulary_{tokenizer}.pkl", "rb") as f:
        vocab = pickle.load(f)
    return data, vocab


def load_word2vec(use_fewer_dimensions=False):
    # TODO: Tog bort smaller vector här. Tänker att vi ändå vill köra samma när vi skapar resultat
    # if use_fewer_dimensions:
    #     model_path = "word_embeddings/word2vec_80.model"
    #     word2vec_path = "word_embeddings/word_vectors_80.npy"
    # else:
    #     model_path = "word_embeddings/word2vec.model"
    #     word2vec_path = "word_embeddings/word_vectors.npy"
    word2vec_path = "token_data/text_vector.npy"

    # The gensim model.
    model = Word2Vec.load("token_data/vector.model")

    # K x N numpy array where K is the number of features for a word and N is the number of words in the corpus.
    word2vec = np.load("token_data/word_vectors.npy")

    # List of all words that appear in chronological order.
    words = np.load("token_data/words.npy")

    # Set of all unique words in the corpus. Is actually list(set) so it supports indexing.
    word_set = np.load("token_data/word_set.npy")

    # Keypairs of words in word_set and their index in word_set
    with open("token_data/vocabulary_vector.pkl", "rb") as f:
        word_to_index = pickle.load(f)

    return model, word2vec, words, word_set, word_to_index


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


def decode(ids, vocab):
    # given ids (list of integers), return Python string
    if config['tokenizer'] == 'char':
        return "".join([vocab[idx] for idx in ids])
    else:
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text


def visualize_tokens(token_indices, vocab):
    output_bytes = [vocab[idx] for idx in token_indices]
    output_bytes = list(map(lambda x: x.decode(
        "utf-8", errors="replace"), output_bytes))
    print(output_bytes)


# Set seed
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize data
with open(sys.argv[1], 'r') as f:
    config = json.load(f)

if config['tokenizer'] == 'vector':
    model, word2vec, words, word_set, word_to_index = load_word2vec(
        use_fewer_dimensions=False)
    test_word2vec_seq(model=model, word2vec_data=word2vec, words=words)
    K = word2vec.shape[0]
    num_words = word2vec.shape[1]
    output_size = word_set.shape[0]
else:
    data, vocab = load_data(config['tokenizer'])
    # Initialize model
    K = len(vocab.keys())
    num_words = len(data)

rnn = RNN(K, config['m'], K).to(device)

criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.RMSprop(rnn.parameters(), lr=config['learning_rate'])

smooth_loss = None

iteration = 0

while True:
    hidden = rnn.initHidden()
    for i in range(0, num_words - config['seq_length'], config['seq_length']):
        if config['tokenizer'] == 'vector':
            X = word2vec[:, i:i+25].T
            X = torch.tensor(X)
            Y = construct_Y_batch(
                start_index=i+1, 
                end_index=i+26, 
                words=words,       
                word_set=word_set, 
                seq_length=config['seq_length'],
                word_to_index=word_to_index
            )
        else:
            X_inds = data[i:i+config['seq_length']]
            Y_inds = data[i+1:i+config['seq_length']+1]

            X = torch.zeros((config['seq_length'], K), dtype=float)
            Y = torch.zeros((config['seq_length'], K), dtype=float)

            # One-hot encode inputs and outputs
            for t, (x_char, y_char) in enumerate(zip(X_inds, Y_inds)):
                X[t, x_char] = 1
                Y[t, y_char] = 1

        # Forward pass
        hidden = rnn.initHidden()
        output, hidden = rnn(X, hidden)

        # Backward pass
        loss = criterion(output, Y)

        # TODO: Undersök detta. Tror konsekvensen är att vi inte backpropagar genom hidden state.
        hidden = hidden.detach()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            rnn.parameters(), config['gradient_clip'])
        optimizer.step()

        if smooth_loss is None:
            smooth_loss = loss
        else:
            smooth_loss = config['smooth_loss_factor'] * \
                smooth_loss + (1 - config['smooth_loss_factor']) * loss

        if (iteration) % config['log_every'] == 0:
            print(f'Iter {iteration:7d}. Smooth loss {
                smooth_loss:7.2f}. Loss {loss:7.2f}.')

        if iteration % config['syntesize_every'] == 0:
            x0 = X[0, :].reshape(1, -1)
            hprev = rnn.initHidden()
            Y_synth = synthesize(rnn, hprev, x0, 200)
            print(decode([torch.argmax(y).item()
                          for y in Y_synth], vocab))

        if iteration >= config['n_iters']:
            break

        iteration += 1

    if iteration >= config['n_iters']:
        break
