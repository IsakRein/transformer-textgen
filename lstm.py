import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size,hidden_size,num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size,output_size)
       
        

    def forward(self, input, hidden):
        out, hidden = self.lstm(input, hidden)
        out = self.linear(out)
        
        return out, hidden
    
    def init_hidden(self, num_layers, batch_size):
        hidden = (torch.zeros((num_layers, batch_size, self.hidden_size)), torch.zeros((num_layers, batch_size, self.hidden_size)))
        return hidden


def load_data(filepath):
    with open(filepath, 'r') as f:
        data = f.read()
    return np.array(list(data))


def synthesize(model, hprev, x0, n):
    h_t = hprev
    x_t = x0
    
    Y = torch.zeros((n, model.input_size), dtype=float)
    
    for t in range(n):
        output, h_t = model(x_t, h_t)
        p_t = output.detach().numpy()
        p_t = p_t.flatten()
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
        x_t[0, 0, ii] = 1

    return Y

def split_data(book_data):
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1
    split_idx_1 = int(train_size*len(book_data))
    split_idx_2 = int((train_size + val_size) *len(book_data))
    train_data = book_data[:split_idx_1]
    val_data = book_data[split_idx_1:split_idx_2]
    test_data = book_data[split_idx_2:]
    return train_data, val_data, test_data

def get_seq(data, seq_length, K, batch_size, idx, char_to_ind):
    X = torch.zeros((batch_size,seq_length, K), dtype=torch.float32)
    Y = torch.zeros((batch_size,seq_length, K), dtype=torch.float32)

    for batch in range(batch_size):
        X_chars = data[idx:idx+seq_length]
        Y_chars = data[idx+1:idx+seq_length+1]
        # One-hot encode inputs and outputs
        for t, (x_char, y_char) in enumerate(zip(X_chars, Y_chars)):
            X[batch,t, char_to_ind[x_char]] = 1
            Y[batch,t, char_to_ind[y_char]] = 1

    return X, Y

def get_loss(output, Y, batch_size, seq_length, K, criterion):
    unbatched_output = torch.reshape(output, (batch_size * seq_length,K))
    unbatched_output_Y = torch.reshape(Y, (batch_size * seq_length,K))
    # Backward pass
    loss = criterion(unbatched_output, unbatched_output_Y)
    loss = torch.sum(loss) / batch_size
    return loss


def train_model():
    torch.manual_seed(0)
    book_data = load_data('data/goblet_book.txt')
    book_chars = np.unique(book_data)
    char_to_ind = {char: idx for idx, char in enumerate(book_chars)}
    ind_to_char = {idx: char for idx,
                   char in enumerate(book_chars)}

    train_data, val_data, test_data = split_data(book_data)

    K = len(book_chars)

    # TODO: Behöver en lägre lr än i min egen implementation.
    # Gissar att RMSprop är annorlunda på något sätt. Den lr för min egen implementation var 0.001.
    m = 100
    eta = 0.001
    gamma = 0.9
    seq_length = 25
    num_layers = 2
    model = CharLSTM(K, m, K, num_layers)

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

    temp = 0
    batch_size = 3

    torch.rand
    while epoch <= 3:
        print(f'-------------')
        print(f'Epoch {epoch}')

        model.zero_grad()
        hidden = model.init_hidden(num_layers, batch_size)
        
        for i in range(0, len(train_data) - (seq_length + batch_size - 1), seq_length + batch_size - 1):
            # Prepare inputs and targets
            
            X, Y = get_seq(train_data, seq_length, K, batch_size, i, char_to_ind)

            # Forward pass
            output, hidden = model(X, hidden)
            
            loss = get_loss(output, Y, batch_size, seq_length, K, criterion)

            # get validation loss
            max_val_idx = len(val_data) - (seq_length + batch_size - 1)
            X_val, Y_val = get_seq(val_data, seq_length, K, batch_size,np.random.randint(max_val_idx),char_to_ind)
            hiddden_val = model.init_hidden(num_layers, batch_size)
            output_val, _ = model(X_val, hiddden_val)
            val_loss = get_loss(output_val, Y_val, batch_size, seq_length, K, criterion)

            # print(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()
            hidden = (hidden[0].detach(), hidden[1].detach())
            

            if smooth_loss is None:
                smooth_loss = loss
                smooth_val_loss = val_loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                smooth_val_loss = 0.999 * smooth_val_loss + 0.001 * val_loss

            if (iteration) % 100 == 0:
                losses.append(loss)
                val_losses.append(val_loss)
                iterations.append(iteration)

            if (iteration) % 1000 == 0:
                print(f'Iter {iteration:7d}. Smooth loss {smooth_loss:7.2f}. Loss {loss:7.2f}. Smooth_val_loss {smooth_val_loss:7.2f}'  )

            if iteration % 10000 == 0:
               
                x0 = torch.zeros(1,1,K)
                x0[0,0,np.random.randint(K)] = 1
                hprev = model.init_hidden(num_layers, 1)
                Y_synth = synthesize(model, hprev, x0, 200)

                txt = ''.join([ind_to_char[torch.argmax(y).item()]
                              for y in Y_synth])

                print(txt)

            iteration += 1
        epoch += 1

    return smooth_loss


if __name__ == '__main__':
    train_model()
