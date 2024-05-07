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
    
    def init_hidden(self):
        hidden = (torch.zeros((1, 1, self.hidden_size)), torch.zeros((1, 1, self.hidden_size)))
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

def train_model():
    torch.manual_seed(0)
    book_data = load_data('data/goblet_book.txt')
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
    num_layers = 1
    model = CharLSTM(K, m, K, num_layers)

    criterion = nn.CrossEntropyLoss(reduction='sum')

    # TODO: Undersök om detta är korrekt.
    optimizer = optim.RMSprop(model.parameters(), lr=eta)

    smooth_loss = None

    losses = []
    iterations = []

    iteration = 0
    epoch = 1

    temp = 0
    batch_size = 1

    torch.rand
    while epoch <= 3:
        print(f'-------------')
        print(f'Epoch {epoch}')

        model.zero_grad()
        hidden = model.init_hidden()
        
        
        for i in range(0, len(book_data) - seq_length, seq_length):
            # Prepare inputs and targets
            X_chars = book_data[i:i+seq_length]
            Y_chars = book_data[i+1:i+seq_length+1]

            X = torch.zeros((1,seq_length, K), dtype=torch.float32)
            Y = torch.zeros((seq_length, K), dtype=torch.float32)

            # One-hot encode inputs and outputs
            # TODO make compatible with larger batches?
            for t, (x_char, y_char) in enumerate(zip(X_chars, Y_chars)):
                X[0,t, char_to_ind[x_char]] = 1
                Y[t, char_to_ind[y_char]] = 1

            # Forward pass
            output, hidden = model(X, hidden)
            unbatched_output = torch.reshape(output, (seq_length,K))
            
            # Backward pass
            loss = criterion(unbatched_output, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()
            hidden = (hidden[0].detach(), hidden[1].detach())
    

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
               
                x0 = torch.zeros(1,1,K)
                x0[0,0,np.random.randint(K)] = 1
                hprev = model.init_hidden()
                Y_synth = synthesize(model, hprev, x0, 200)

                txt = ''.join([ind_to_char[torch.argmax(y).item()]
                              for y in Y_synth])

                print(txt)

            iteration += 1
        epoch += 1

    return smooth_loss


if __name__ == '__main__':
    train_model()
