import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
import pickle
import sys
import json
from gensim.models import Word2Vec


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size))

        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        h, c = hidden

        i = torch.sigmoid(x @ self.W_ii + h @ self.W_hi + self.b_i)
        f = torch.sigmoid(x @ self.W_if + h @ self.W_hf + self.b_f)
        g = torch.tanh(x @ self.W_ig + h @ self.W_hg + self.b_g)
        o = torch.sigmoid(x @ self.W_io + h @ self.W_ho + self.b_o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = nn.ModuleList(
            [LSTMCell(input_size if i == 0 else hidden_size,
                            hidden_size) for i in range(num_layers)]
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        h, c = hidden, cell
        outputs = []
        for x in input.split(1, dim=1):  # Split the input along the sequence dimension
            x = x.squeeze(1)
            for i, lstm_cell in enumerate(self.lstm_cells):
                h[i], c[i] = lstm_cell(x, (h[i], c[i]))
                x = h[i]
            outputs.append(x.unsqueeze(1))
        lstm_out = torch.cat(outputs, dim=1)
        output = self.fc(lstm_out)
        return output, h, c

    def initHidden(self, batch_size=1):
        hidden = [torch.zeros(batch_size, self.hidden_size)
                  for _ in range(self.num_layers)]
        cell = [torch.zeros(batch_size, self.hidden_size)
                for _ in range(self.num_layers)]
        return hidden, cell


def load_data(tokenizer):
    data = torch.tensor(
        np.load(f"token_data/text_{tokenizer}.npy"), dtype=torch.long)
    with open(f"token_data/vocabulary_{tokenizer}.pkl", "rb") as f:
        vocab = pickle.load(f)
    return data, vocab


def load_word2vec(use_fewer_dimensions=False):
    word2vec_path = "token_data/text_vector.npy"
    model = Word2Vec.load("token_data/vector.model")
    word2vec = np.load("token_data/word_vectors.npy")
    words = np.load("token_data/words.npy")
    word_set = np.load("token_data/word_set.npy")
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


def synthesize(lstm, hprev, cprev, x0, n):
    h_t, c_t = hprev, cprev
    x_t = x0
    Y = []

    for t in range(n):
        output, h_t, c_t = lstm(x_t.unsqueeze(0), h_t, c_t)
        p_t = torch.softmax(output.squeeze(0), dim=1).detach().cpu().numpy()
        p_t = p_t.flatten()  # Flatten to 1D array
        p_t = p_t / np.sum(p_t)

        # generate x_t
        ix = np.random.choice(len(p_t), p=p_t)
        Y.append(ix)

        # Update x_t for next iteration
        x_t = torch.zeros_like(x0)
        x_t[0, ix] = 1

    return Y


def decode(ids, vocab):
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
    K = len(vocab.keys())
    num_words = len(data)

lstm = LSTM(K, config['m'], K, num_layers=config.get(
    'num_layers', 1)).to(device)

criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.RMSprop(lstm.parameters(), lr=config['learning_rate'])

smooth_loss = None
iteration = 0

while True:
    hidden, cell = lstm.initHidden(batch_size=1)
    for i in range(0, num_words - config['seq_length'], config['seq_length']):
        if config['tokenizer'] == 'vector':
            X = word2vec[:, i:i+25].T
            X = torch.tensor(X, device=device)
            Y = construct_Y_batch(
                start_index=i+1,
                end_index=i+26,
                words=words,
                word_set=word_set,
                seq_length=config['seq_length'],
                word_to_index=word_to_index
            ).to(device)
        else:
            X_inds = data[i:i+config['seq_length']]
            Y_inds = data[i+1:i+config['seq_length']+1]

            X = torch.zeros((config['seq_length'], K),
                            dtype=torch.float, device=device)
            Y = torch.zeros((config['seq_length'], K),
                            dtype=torch.float, device=device)

            for t, (x_char, y_char) in enumerate(zip(X_inds, Y_inds)):
                X[t, x_char] = 1
                Y[t, y_char] = 1

        hidden, cell = lstm.initHidden(batch_size=1)
        output, hidden, cell = lstm(X.unsqueeze(0), hidden, cell)

        loss = criterion(output.squeeze(0), Y)

        hidden = [h.detach() for h in hidden]
        cell = [c.detach() for c in cell]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            lstm.parameters(), config['gradient_clip'])
        optimizer.step()

        if smooth_loss is None:
            smooth_loss = loss
        else:
            smooth_loss = config['smooth_loss_factor'] * \
                smooth_loss + (1 - config['smooth_loss_factor']) * loss

        if (iteration) % config['log_every'] == 0:
            print(f'Iter {iteration:7d}. Smooth loss {
                  smooth_loss:7.2f}. Loss {loss:7.2f}.')

        if iteration % config['synthesize_every'] == 0:
            x0 = X[0, :].reshape(1, -1)
            hprev, cprev = lstm.initHidden(batch_size=1)
            Y_synth = synthesize(lstm, hprev, cprev, x0, 200)
            print(decode(Y_synth, vocab))

        if iteration >= config['n_iters']:
            break

        iteration += 1

    if iteration >= config['n_iters']:
        break
