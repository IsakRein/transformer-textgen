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
            start_idx = np.random.randint(len(train_data))-config['eval_iters']
        elif split == "val":
            start_idx = np.random.randint(len(val_data))-config['eval_iters']
        for k in range(config['eval_iters']):
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

def load_spell_checker(input_text):
    spell_checker = SpellChecker(language=None)
    known = re.sub(r'[^a-zA-Z\s]', ' ', input_text).split()
    spell_checker.word_frequency.load_words(known)
    return spell_checker

def evaluate_spelling(spell_checker, generated_text):
    words = re.sub(r'[^a-zA-Z\s]', ' ', generated_text).split()
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
    n = int(len(data) * config['train_size'])
    train_data = data[:n]
    val_data = data[n:]
    K = len(vocab.keys())
    num_words = len(train_data)

model = RNN(K, config['m'], K).to(device)

criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'])

smooth_loss = None

iteration = 0

test_file = re.findall(r'\.\/tests\/(\w+)\.json', sys.argv[1])[0]
PATH = f"./model_data/{test_file}"
model_loaded, train_loss_values, val_loss_values , train_perplexity, val_perplexity = load_model(PATH)

if (not model_loaded):
    while True:
        hidden = model.initHidden()
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
spell_checker = load_spell_checker(text)
spelling_accuracy = evaluate_spelling(spell_checker, sample)
with open (f"{PATH}/spelling_accuracy.txt", "w") as file:
    file.write(str(spelling_accuracy))