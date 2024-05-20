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
import time


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        out, hidden = self.lstm(input, hidden)
        out = self.linear(out)

        return out, hidden

    def init_hidden(self, num_layers, batch_size):
        hidden = (torch.zeros((num_layers, batch_size, self.hidden_size), device=device),
                  torch.zeros((num_layers, batch_size, self.hidden_size), device=device))
        return hidden


def load_data(filepath):
    with open(filepath, 'r') as f:
        data = f.read()
    return np.array(list(data))


def synthesize(model, hprev, x0, n, temperature):
    h_t = hprev
    x_t = x0.to(device)  # Make sure x0 is on the right device

    Y = torch.zeros((n, model.input_size), dtype=torch.float, device=device)

    for t in range(n):
        output, h_t = model(x_t, h_t)
        # Move tensor to CPU for numpy operations
        p_t = output.squeeze().detach().cpu().numpy()
        p_t = np.exp(p_t / temperature)
        p_t /= np.sum(p_t)

        # Random sampling
        i = np.random.choice(len(p_t), p=p_t)

        # Update Y and x_t for next iteration
        Y[t, i] = 1
        x_t = torch.zeros_like(x0, device=device)
        x_t[0, 0, i] = 1

    return Y


def split_data(book_data):
    train_size = config['train_size']
    val_size = config['val_size']
    test_size = config['test_size']
    split_idx_1 = int(train_size*len(book_data))
    split_idx_2 = int((train_size + val_size) * len(book_data))
    train_data = book_data[:split_idx_1]
    val_data = book_data[split_idx_1:split_idx_2]
    test_data = book_data[split_idx_2:]
    return train_data, val_data, test_data


def get_seq(data, seq_length, K, batch_size, idx, char_to_ind):
    X = torch.zeros((batch_size, config['seq_length'], K),
                    dtype=torch.float32, device=device)
    Y = torch.zeros((batch_size, seq_length, K),
                    dtype=torch.float32, device=device)

    for batch in range(batch_size):
        X_chars = data[idx:idx+seq_length]
        Y_chars = data[idx+1:idx+seq_length+1]
        # One-hot encode inputs and outputs
        for t, (x_char, y_char) in enumerate(zip(X_chars, Y_chars)):
            X[batch, t, char_to_ind[x_char]] = 1
            Y[batch, t, char_to_ind[y_char]] = 1

    X.to(device)
    Y.to(device)

    return X, Y


def get_loss(output, Y, batch_size, seq_length, K, criterion):
    unbatched_output = torch.reshape(output, (batch_size * seq_length, K))
    unbatched_output_Y = torch.reshape(Y, (batch_size * seq_length, K))
    # Backward pass
    loss = criterion(unbatched_output, unbatched_output_Y)
    loss = torch.sum(loss) / batch_size / seq_length
    return loss


# Set seed
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize data
with open(sys.argv[1], 'r') as f:
    config = json.load(f)

# TODO: Rewrite so that all encodings are supported
start_time = time.time()
book_data = load_data('./data/goblet_book.txt')
book_chars = np.unique(book_data)
char_to_ind = {char: idx for idx, char in enumerate(book_chars)}
ind_to_char = {idx: char for idx,
               char in enumerate(book_chars)}

train_data, val_data, test_data = split_data(book_data)

K = len(book_chars)

# TODO: Behöver en lägre lr än i min egen implementation.
# Gissar att RMSprop är annorlunda på något sätt. Den lr för min egen implementation var 0.001.

model = CharLSTM(K, config['hidden_layer_size'], K,
                 config['num_layers']).to(device)

criterion = nn.CrossEntropyLoss(reduction='none')

# TODO: Undersök om detta är korrekt.
optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'])

smooth_loss = None
smooth_val_loss = None

losses = []
val_losses = []
iterations = []

iteration = 0

while True:
    model.zero_grad()
    hidden = model.init_hidden(config['num_layers'], config['batch_size'])

    for i in range(0, len(train_data) - (config['seq_length'] + config['batch_size'] - 1), config['seq_length'] + config['batch_size'] - 1):
        # Prepare inputs and targets

        X, Y = get_seq(train_data, config['seq_length'], K,
                       config['batch_size'], i, char_to_ind)

        # Forward pass
        output, hidden = model(X, hidden)

        loss = get_loss(
            output, Y, config['batch_size'], config['seq_length'], K, criterion)

        # get validation loss
        max_val_idx = len(val_data) - \
            (config['seq_length'] + config['batch_size'] - 1)
        X_val, Y_val = get_seq(val_data, config['seq_length'], K, config['batch_size'], np.random.randint(
            max_val_idx), char_to_ind)
        hiddden_val = model.init_hidden(
            config['num_layers'], config['batch_size'])
        output_val, _ = model(X_val, hiddden_val)
        val_loss = get_loss(output_val, Y_val, config['batch_size'],
                            config['seq_length'], K, criterion)

        # print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config['gradient_clip'])
        optimizer.step()
        hidden = tuple([each.data for each in hidden])

        if smooth_loss is None:
            smooth_loss = loss
            smooth_val_loss = val_loss
        else:
            smooth_loss = config['smooth_loss_factor'] * \
                smooth_loss + (1 - config['smooth_loss_factor']) * loss
            smooth_val_loss = config['smooth_loss_factor'] * \
                smooth_val_loss + (1 - config['smooth_loss_factor']) * val_loss

        if (iteration) % config['log_every'] == 0:
            losses.append(smooth_loss)
            val_losses.append(smooth_val_loss)
            iterations.append(iteration)
            print(f'Iter {iteration:7d}. Smooth loss {smooth_loss:7.2f}. Loss {
                  loss:7.2f}. Smooth val loss {smooth_val_loss:7.2f}. Val loss {val_loss:7.2f}.')

        if iteration % config['syntesize_every'] == 0:
            x0 = torch.zeros(1, 1, K)
            x0[0, 0, np.random.randint(K)] = 1
            hprev = model.init_hidden(config['num_layers'], 1)
            Y_synth = synthesize(model, hprev, x0, 200, config['temperature'])

            txt = ''.join([ind_to_char[torch.argmax(y).item()]
                           for y in Y_synth])

            # print(txt)

        if iteration >= config['n_iters']:
            break

        iteration += 1

    if iteration >= config['n_iters']:
        break
