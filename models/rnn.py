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
import os
import re
from torcheval.metrics.text import Perplexity
from spellchecker import SpellChecker


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
        seq_length, K = input.shape
        output = torch.zeros((seq_length, self.output_size), dtype=torch.float)
        for i in range(seq_length):
            hidden = torch.tanh(self.U(input[i]) + self.W(hidden))
            output[i, :] = torch.log_softmax(self.V(hidden), dim=1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, dtype=float)


def load_data(tokenizer):
    data = torch.tensor(
        np.load(f"token_data/train_{tokenizer}.npy"), dtype=torch.long)
    with open(f"token_data/train_vocabulary_{tokenizer}.pkl", "rb") as f:
        vocab = pickle.load(f)
    return data, vocab


def load_word2vec():
    # The gensim model.
    model = Word2Vec.load("token_data/vec_gensim.model")

    # K x N numpy array where K is the number of features for a word and N is the number of words in the corpus.
    word2vec = np.load("token_data/text_vec.npy")

    # List of all words that appear in chronological order.
    words = np.load("token_data/vec_words.npy")

    # Dict mapping words to indices
    with open("token_data/word_to_index_vec.pkl", "rb") as f:
        word_to_index = pickle.load(f)

    # Keypairs of words in word_set and their index in word_set
    with open("token_data/vocabulary_vec.pkl", "rb") as f:
        vocab = pickle.load(f)

    return model, word2vec, words, word_to_index, vocab

def test_word2vec_seq(model, word2vec_data, words):
    for i in range(len(words)):
        assert np.array_equal(model.wv[words[i]], word2vec_data[:, i])

def synthesize_word2vec(rnn, hprev, x0, n, vocab, word2vec_model):
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
        word_vec = word2vec_model.wv[vocab[ii]]

        # Update Y and x_t for next iteration
        Y[t] = vocab[ii]

        x_t = torch.reshape(torch.tensor(word_vec), x_t.shape)
        x_t = x_t.double()
    return Y

def nucleus_sampling(rnn, h, x, theta, max_new_tokens):

        h_t = h
        x_t = x
        Y = torch.zeros((max_new_tokens), dtype=float)
        
        for t in range(max_new_tokens):
            output, h_t = rnn(x_t, h_t)
            probs = torch.nn.functional.softmax(output, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            cutoff_index = torch.where(cumulative_probs[0] >= theta)[0][0] + 1
            
            sorted_probs[0][cutoff_index:] = 0
            sorted_probs = sorted_probs / torch.sum(sorted_probs)
            
            next_token = torch.multinomial(sorted_probs, num_samples=1)

            x_t = torch.zeros_like(x0)
            x_t[0, sorted_indices[0][next_token][0].item()] = 1
        
            Y[t] = sorted_indices[0][next_token][0].item()
    
        return Y.tolist()

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

def softmax(x, temperature=1):
    return np.exp(x / temperature) / (np.sum(np.exp(x / temperature), axis=0) + 1e-15)

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

def load_model(PATH):
    if (os.path.exists(PATH)):
        print("Loading model")
        model.load_state_dict(torch.load(f"{PATH}/model.pth"))
        train_loss_values = torch.load(f"{PATH}/train_losses.pth")
        val_loss_values = torch.load(f"{PATH}/val_losses.pth")
        train_perplexity = torch.load(f"{PATH}/train_perplexity.pth")
        val_perplexity = torch.load(f"{PATH}/val_perplexity.pth")
        return True, train_loss_values, val_loss_values , train_perplexity, val_perplexity
    else:
        return False, [], [], [], []
    
def save_model(PATH,train_loss_values,val_loss_values,train_perplexity, val_perplexity):
    os.mkdir(PATH) 
    torch.save(model.state_dict(), f"{PATH}/model.pth")
    torch.save(torch.tensor(train_loss_values), f"{PATH}/train_losses.pth")
    torch.save(torch.tensor(val_loss_values), f"{PATH}/val_losses.pth")
    torch.save(torch.tensor(train_perplexity), f"{PATH}/train_perplexity.pth")
    torch.save(torch.tensor(val_perplexity), f"{PATH}/val_perplexity.pth")


@torch.no_grad()
def estimate_metrics():
    out = {}
    perplexity = {}
    model.eval()
    hidden = model.initHidden()
    
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        perplexity_metric = Perplexity()
        if split == "train":
            if config['tokenizer'] == 'vec':
                start_idx = np.random.randint((train_data.shape[1])) - config['eval_iters']
            else: 
                start_idx = np.random.randint(len(train_data))-config['eval_iters']
        elif split == "val":
            if config['tokenizer'] == 'vec':
                start_idx = np.random.randint((val_data.shape[1])) - config['eval_iters']
            else:
                start_idx = np.random.randint(len(val_data))-config['eval_iters']
        for k in range(config['eval_iters']):
            if config['tokenizer'] == 'vec':
                X, Y = construct_word2Vec_batch(split, start_idx + k)
            else:
                X, Y = get_batch(split,start_idx + k)
            output, hidden = model(X, hidden)
            loss = criterion(output, Y)
            losses[k] = loss.item()
            labels = torch.argmax(Y, dim=1)
            perplexity_metric.update(output.unsqueeze(0), labels.unsqueeze(0))
        out[split] = losses.mean().item()
        perplexity[split] = perplexity_metric.compute().item()
    model.train()
    return out, perplexity

def construct_word2Vec_batch(split, i):
    data = train_data if split == 'train' else val_data
    X = data[:, i:i+config['seq_length']].T
    X = torch.tensor(X)
    Y = torch.zeros((config['seq_length'], output_size))

    k = 0
    for j in range(i+1, i + config['seq_length'] + 1):
        Y_index = word_to_index[words[j]]
        Y[k, Y_index] = 1 
        k += 1 
    return X, Y

def get_batch(split, i):
    data = train_data if split == 'train' else val_data
    X_inds = data[i:i+config['seq_length']]
    Y_inds = data[i+1:i+config['seq_length']+1]

    X = torch.zeros((config['seq_length'], K), dtype=float)
    Y = torch.zeros((config['seq_length'], K), dtype=float)

    # One-hot encode inputs and outputs
    for t, (x_char, y_char) in enumerate(zip(X_inds, Y_inds)):
        X[t, x_char] = 1
        Y[t, y_char] = 1
    return X,Y



def evaluate_spelling(spell_checker, generated_text):
    words = re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", generated_text)
    misspelled = spell_checker.unknown(words)
    total_words = len(words)
    correctly_spelled_words = total_words - len(misspelled)
    correctly_spelled_percentage = (correctly_spelled_words / total_words) * 100
    return correctly_spelled_percentage

# Set seed
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize data
with open(sys.argv[1], 'r') as f:
    config = json.load(f)

if config['tokenizer'] == 'vec':
    word2vec_model, data, words, word_to_index, vocab = load_word2vec()
    test_word2vec_seq(model=word2vec_model, word2vec_data=data, words=words)
    K = data.shape[0]
    n = int(data.shape[1] * config['train_size'])
    train_data = data[:, :n]
    val_data = data[:, :n]
else:
    data, vocab = load_data(config['tokenizer'])
    K = len(vocab.keys())
    n = int(len(data) * config['train_size'])
    train_data = data[:n]
    val_data = data[n:]

output_size = len(vocab.keys())
if config['tokenizer'] == 'vec':
    num_words = data.shape[1]
else:
    num_words = len(data)

model = RNN(K, config['m'], output_size).to(device)

criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'])

smooth_loss = None

iteration = 0

test_files = re.findall(r'tests\/(\w+)\.json', sys.argv[1])
if test_files != []:
    test_file = test_files[0]
    PATH = f"./model_data/{test_file}"
else:
    PATH = ""
model_loaded, train_loss_values, val_loss_values , train_perplexity, val_perplexity = load_model(PATH)

if (not model_loaded):
    while True:
        hidden = model.initHidden()
        for i in range(0, num_words - config['seq_length'], config['seq_length']):
            if config['tokenizer'] == 'vec':
                X, Y = construct_word2Vec_batch("train", i)
            else:
                X,Y = get_batch("train",i)
                
            # Forward pass
            hidden = model.initHidden()
            output, hidden = model(X, hidden)

            # Backward pass
            loss = criterion(output, Y)

            # TODO: Undersök detta. Tror konsekvensen är att vi inte backpropagar genom hidden state.
            hidden = hidden.detach()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['gradient_clip'])
            optimizer.step()

            if (iteration) % config['log_every'] == 0:
                losses, perplexity = estimate_metrics()
                train_loss_values.append(losses['train'])
                val_loss_values.append(losses['val'])
                train_perplexity.append(perplexity['train'])
                val_perplexity.append(perplexity['val'])
                print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train perplexity {perplexity['train']:.4f}, val perplexity {perplexity['val']:.4f}")

            if iteration % config['syntesize_every'] == 0:
                x0 = X[0, :].reshape(1, -1)
                hprev = model.initHidden()
                if config['tokenizer'] == 'vec':
                    generated_text = synthesize_word2vec(model, hprev, x0, 200, vocab, word2vec_model)
                    print("".join(generated_text))
                else:
                    Y_synth = synthesize(model, hprev, x0, 200)
                    print(decode([torch.argmax(y).item()
                                for y in Y_synth], vocab))

            if iteration >= config['n_iters']:
                break

            iteration += 1
            
        if iteration >= config['n_iters']:
            break

    losses, perplexity = estimate_metrics()
    train_loss_values.append(losses['train'])
    val_loss_values.append(losses['val'])
    train_perplexity.append(perplexity['train'])
    val_perplexity.append(perplexity['val'])
    print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train perplexity {perplexity['train']:.4f}, val perplexity {perplexity['val']:.4f}")
    save_model(PATH, train_loss_values, val_loss_values, train_perplexity, val_perplexity)



print(f'Final train loss: {train_loss_values[-1]:.4f}')
print(f'Final val loss: {val_loss_values[-1]:.4f}')
print(f'Final train perplexity: {train_perplexity[-1]:.4f}')
print(f'Final val perplexity: {val_perplexity[-1]:.4f}')

# Generate text
print(f'Synthesizing text...')

hprev = model.initHidden()
X = torch.zeros((config['seq_length'], K), dtype=float)
X[0,0] = 1
x0 = X[0, :].reshape(1, -1)

if config['sampling'] == "temp":
    if config['tokenizer'] == 'vec':
        generated_text = synthesize_word2vec(model, hprev, x0, 200, vocab, word2vec_model)
        print("".join(generated_text))
    else:
        Y_synth = synthesize(model, hprev, x0, 200)
        sample = [torch.argmax(y).item() for y in Y_synth]
        sample = decode(sample, vocab)
    
elif config['sampling'] == "nucleus":
    Y_synth = nucleus_sampling(model, hprev, x0, theta=config['nucleus'], max_new_tokens=config['max_new_tokens'])
    sample = decode(Y_synth, vocab)
    
print(sample)
with open (f"{PATH}/text_sample.txt", "w") as file:
    file.write(sample)

with open('./data/goblet_book.txt', 'r', encoding='utf-8') as f:
    text = f.read()
spell_checker = SpellChecker()
spelling_accuracy = evaluate_spelling(spell_checker, sample)
with open (f"{PATH}/spelling_accuracy.txt", "w") as file:
    file.write(str(spelling_accuracy))