import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
import time

# Setting device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


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
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1
    split_idx_1 = int(train_size*len(book_data))
    split_idx_2 = int((train_size + val_size) * len(book_data))
    train_data = book_data[:split_idx_1]
    val_data = book_data[split_idx_1:split_idx_2]
    test_data = book_data[split_idx_2:]
    return train_data, val_data, test_data


def get_seq(data, seq_length, K, batch_size, idx, char_to_ind):
    X = torch.zeros((batch_size, seq_length, K),
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
    loss = torch.sum(loss) / batch_size
    return loss


def train_model(eta, batch_size, num_layers, hidden_layer_size, temperature):
    torch.manual_seed(0)
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

    gamma = 0.9
    seq_length = 25

    model = CharLSTM(K, hidden_layer_size, K, num_layers).to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')

    # TODO: Undersök om detta är korrekt.
    optimizer = optim.RMSprop(model.parameters(), lr=eta)

    smooth_loss = None
    smooth_val_loss = None

    losses = []
    val_losses = []
    iterations = []

    iteration = 0
    epoch = 1

    torch.rand
    while epoch <= 3:
        print(f'-------------')
        print(f'Epoch {epoch}')

        model.zero_grad()
        hidden = model.init_hidden(num_layers, batch_size)

        for i in range(0, len(train_data) - (seq_length + batch_size - 1), seq_length + batch_size - 1):
            # Prepare inputs and targets

            X, Y = get_seq(train_data, seq_length, K,
                           batch_size, i, char_to_ind)

            # Forward pass
            output, hidden = model(X, hidden)

            loss = get_loss(output, Y, batch_size, seq_length, K, criterion)

            # get validation loss
            max_val_idx = len(val_data) - (seq_length + batch_size - 1)
            X_val, Y_val = get_seq(val_data, seq_length, K, batch_size, np.random.randint(
                max_val_idx), char_to_ind)
            hiddden_val = model.init_hidden(num_layers, batch_size)
            output_val, _ = model(X_val, hiddden_val)
            val_loss = get_loss(output_val, Y_val, batch_size,
                                seq_length, K, criterion)

            # print(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()
            hidden = tuple([each.data for each in hidden])

            if smooth_loss is None:
                smooth_loss = loss
                smooth_val_loss = val_loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                smooth_val_loss = 0.999 * smooth_val_loss + 0.001 * val_loss

            if (iteration) % 100 == 0:
                losses.append(smooth_loss)
                val_losses.append(smooth_val_loss)
                iterations.append(iteration)

            if (iteration) % 1000 == 0:
                deltaTime = time.time() - start_time
                hours = int(deltaTime // 3600)
                minutes = int((deltaTime % 3600) // 60)
                seconds = int(deltaTime % 60)
                time_str = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
                print(f'[{time_str}] Iter {iteration:7d}. Smooth loss {smooth_loss:7.2f}. Loss {loss:7.2f}. Smooth_val_loss {smooth_val_loss:7.2f}')

            if iteration % 10000 == 0:

                x0 = torch.zeros(1, 1, K)
                x0[0, 0, np.random.randint(K)] = 1
                hprev = model.init_hidden(num_layers, 1)
                Y_synth = synthesize(model, hprev, x0, 200, temperature)

                txt = ''.join([ind_to_char[torch.argmax(y).item()]
                              for y in Y_synth])

                # print(txt)

            iteration += 1
        epoch += 1

    return losses, val_losses


if __name__ == '__main__':
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [1, 64, 128]
    num_layers = [1, 2]
    hidden_layer_size = [64, 128, 256]
    temperatures = [0.3, 0.6, 1]

    # default values
    eta = 0.001
    batch_size = 1
    layers = 1
    m = 100
    temp = 1

    # Train an RNN baseline on the dataset you use and compare at least to both a one and two layer LSTM both qualitatively and quantitatively
    train_loss_values, val_loss_values = train_model(
        eta=eta, batch_size=batch_size, num_layers=1, hidden_layer_size=m, temperature=temp)
    np.save("Train_loss_one_layer.npy", np.array(train_loss_values))
    np.save("Val_loss_one_layer.npy", np.array(val_loss_values))
    train_loss_values, val_loss_values = train_model(
        eta=eta, batch_size=batch_size, num_layers=2, hidden_layer_size=m, temperature=temp)
    np.save("Train_loss_two_layers.npy", np.array(train_loss_values))
    np.save("Val_loss_two_layers.npy", np.array(val_loss_values))

    # Investigate how increasing the number of the nodes of the hidden state increases or decreases performance.
    for size in hidden_layer_size:
        train_loss_values, val_loss_values = train_model(
            eta=eta, batch_size=batch_size, num_layers=layers, hidden_layer_size=size, temperature=temp)
        np.save(f"Train loss hidden_size={size}.npy", np.array(train_loss_values))
        np.save(f"Val loss hidden size={size}.npy", np.array(val_loss_values))

    # Investigate the influence of different training parameters such as batch size and learning rate.
    # You can for example perform a grid search or random search to find the optimal training parameters.
    for eta in learning_rates:
        for batch_size in batch_sizes:
            train_loss_values, val_loss_values = train_model(
                eta=eta, batch_size=batch_size, num_layers=layers, hidden_layer_size=size, temperature=temp)
            np.save(f"Train eta={eta} batch_size={batch_size}.npy", np.array(train_loss_values))
            np.save(f"Val eta={eta} batch_size={batch_size}.npy", np.array(val_loss_values))
