import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
import pickle
import sys
import os
import re
from torcheval.metrics.text import Perplexity
from spellchecker import SpellChecker


class TextProcessor:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.char_to_idx[c] for c in s]

    def decode(self, l):
        return ''.join([self.idx_to_char[i] for i in l])

    def tensor_to_text(self, t):
        return ''.join([self.idx_to_char[i.item()] for i in t])


def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(
        len(data) - config['seq_length'], (config['batch_size'],), device=device)
    X_batch = torch.stack([data[j:j+config['seq_length']] for j in idx])
    Y_batch = torch.stack([data[j+1:j+config['seq_length']+1] for j in idx])
    Y_batch = Y_batch.reshape((1, config['batch_size'] * config['seq_length']))
    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
    return X_batch, Y_batch


def load_data(tokenizer):
    data = torch.tensor(
        np.load(f"token_data/train_{tokenizer}.npy"), dtype=torch.long, device=device)
    with open(f"token_data/train_vocabulary_{tokenizer}.pkl", "rb") as f:
        vocab = pickle.load(f)
    val_data = torch.tensor(np.load(
        f"token_data/validation_{tokenizer}.npy"), dtype=torch.long, device=device)
    return data, vocab, val_data


def decode(ids, vocab):
    if config['tokenizer'] == 'char':
        return "".join([vocab[idx.item()] for idx in ids])
    else:
        tokens = b"".join(vocab[idx.item()] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text


@torch.no_grad()
def estimate_metrics():
    out = {}
    perplexity = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'], device=device)
        perplexity_metric = Perplexity().to(device)
        for k in range(config['eval_iters']):
            X, Y = get_batch(split)
            outputs, loss = model(X, Y)
            losses[k] = loss.item()
            perplexity_metric.update(outputs.unsqueeze(0), Y.view(
                1, config['batch_size'] * config['seq_length']))
        out[split] = losses.mean().item()
        perplexity[split] = perplexity_metric.compute().item()
    model.train()
    return out, perplexity


def evaluate_spelling(spell_checker, generated_text):
    words = re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", generated_text)
    misspelled = spell_checker.unknown(words)
    total_words = len(words)
    correctly_spelled_words = total_words - len(misspelled)
    correctly_spelled_percentage = (
        correctly_spelled_words / total_words) * 100
    return correctly_spelled_percentage


class PositionalEncoding(nn.Module):
    def __init__(self, n_embd, seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(seq_length, n_embd, device=device)
        position = torch.arange(
            0, seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float().to(
            device) * -(math.log(10000.0) / n_embd))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config['n_embd'], head_size, bias=False)
        self.query = nn.Linear(config['n_embd'], head_size, bias=False)
        self.value = nn.Linear(config['n_embd'], head_size, bias=False)
        self.dropout = nn.Dropout(config['dropout'])
        self.register_buffer('tril', torch.tril(torch.ones(
            config['seq_length'], config['seq_length'], device=device)))

    def forward(self, X):
        _, seq_length, _ = X.shape
        k, q, v = self.key(X), self.query(X), self.value(X)
        affinity = q @ k.transpose(-2, -1) / (self.key.weight.size(-1) ** 0.5)
        affinity = affinity.masked_fill(
            self.tril[:seq_length, :seq_length] == 0, float('-inf'))
        affinity = F.softmax(affinity, dim=-1)
        affinity = self.dropout(affinity)
        return affinity @ v


class MultiheadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.projection = nn.Linear(config['n_embd'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, X):
        out = torch.cat([head(X) for head in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class AddAndNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(config['n_embd'])

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(
            n_embd, config['feed_forward_multiplier'] * n_embd)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(
            config['feed_forward_multiplier'] * n_embd, n_embd)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, X):
        X = self.linear_1(X)
        X = self.relu(X)
        X = self.linear_2(X)
        return self.dropout(X)


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = config['n_embd'] // config['n_head']
        self.multi_head_attention = MultiheadAttention(
            config['n_head'], head_size)
        self.mlp = FeedForward(config['n_embd'])
        self.add_and_norm_1 = AddAndNorm()
        self.add_and_norm_2 = AddAndNorm()

    def forward(self, X):
        X = self.add_and_norm_1(X, lambda x: self.multi_head_attention(x))
        X = self.add_and_norm_1(X, lambda x: self.mlp(x))
        return X


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, config['n_embd'])
        self.positional_encoding = PositionalEncoding(
            config['n_embd'], config['seq_length'])
        self.decoder_blocks = nn.Sequential(
            *[DecoderBlock() for _ in range(config['n_layer'])])
        self.norm = nn.LayerNorm(config['n_embd'])
        self.transform_to_vocab_size = nn.Linear(config['n_embd'], vocab_size)

    def forward(self, X, Y=None):
        token_embeddings = self.embed_tokens(X)
        X = self.positional_encoding(token_embeddings)
        X = self.decoder_blocks(X)
        X = self.norm(X)
        logits = self.transform_to_vocab_size(X)
        if Y is None:
            loss = None
        else:
            batch_size, seq_length, vocabulary_size = logits.shape
            logits = logits.view(batch_size * seq_length, vocabulary_size)
            Y = Y.view(batch_size * seq_length)
            loss = F.cross_entropy(logits, Y)
        return logits, loss

    def synthesize(self, tokens, max_new_tokens, temperature):
        for _ in range(max_new_tokens):
            input = tokens[:, -config['seq_length']:]

            logits, _ = self(input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens

    def nucleus_sampling(self, tokens, max_new_tokens, theta):
        for _ in range(max_new_tokens):
            input = tokens[:, -config['seq_length']:]
            logits, _ = self(input)

            logits = logits[:, -1, :]

            probs = torch.nn.functional.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            cutoff_index = torch.where(cumulative_probs[0] >= theta)[0][0] + 1

            sorted_probs[0][cutoff_index:] = 0
            sorted_probs = sorted_probs / torch.sum(sorted_probs)

            next_token = torch.multinomial(sorted_probs, num_samples=1)
            tokens = torch.cat((tokens, sorted_indices[0][next_token]), dim=1)
        return tokens


def load_model(PATH):
    if (os.path.exists(PATH)):
        print("Loading model")
        model.load_state_dict(torch.load(
            f"{PATH}/model.pth", map_location=device))
        train_loss_values = torch.load(
            f"{PATH}/train_losses.pth", map_location=device)
        val_loss_values = torch.load(
            f"{PATH}/val_losses.pth", map_location=device)
        train_perplexity = torch.load(
            f"{PATH}/train_perplexity.pth", map_location=device)
        val_perplexity = torch.load(
            f"{PATH}/val_perplexity.pth", map_location=device)
        return True, train_loss_values, val_loss_values, train_perplexity, val_perplexity
    else:
        return False, [], [], [], []


def save_model(PATH, train_loss_values, val_loss_values, train_perplexity, val_perplexity):
    os.mkdir(PATH)
    torch.save(model.state_dict(), f"{PATH}/model.pth")
    torch.save(torch.tensor(train_loss_values), f"{PATH}/train_losses.pth")
    torch.save(torch.tensor(val_loss_values), f"{PATH}/val_losses.pth")
    torch.save(torch.tensor(train_perplexity), f"{PATH}/train_perplexity.pth")
    torch.save(torch.tensor(val_perplexity), f"{PATH}/val_perplexity.pth")


# Set seed
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize data
with open(sys.argv[1], 'r') as f:
    config = json.load(f)

train_data, vocab, val_data = load_data(config['tokenizer'])

n = int(len(train_data) * config['train_size'])

# Initialize model and optimizer
model = DecoderOnlyTransformer(len(vocab)).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config['learning_rate'], weight_decay=config['lambda'])

test_file = re.findall(r'tests\/(\w+)\.json', sys.argv[1])[0]
PATH = f"./model_data/{test_file}"
model_loaded, train_loss_values, val_loss_values, train_perplexity, val_perplexity = load_model(
    PATH)

if (not model_loaded):
    # Training loop
    for iter in range(config['n_iters']):
        X_batch, Y_batch = get_batch('train')
        logits, loss = model(X_batch, Y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % config['log_every'] == 0:
            losses, perplexity = estimate_metrics()
            train_loss_values.append(losses['train'])
            val_loss_values.append(losses['val'])
            train_perplexity.append(perplexity['train'])
            val_perplexity.append(perplexity['val'])
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train perplexity {
                  perplexity['train']:.4f}, val perplexity {perplexity['val']:.4f}")

        if iter % config['syntesize_every'] == 0:
            prompt = torch.tensor([[0]], dtype=torch.long, device=device)
            print(decode(model.synthesize(
                prompt, max_new_tokens=config['max_new_tokens'], temperature=config['temperature'])[0], vocab))

    # final eval
    losses, perplexity = estimate_metrics()
    train_loss_values.append(losses['train'])
    val_loss_values.append(losses['val'])
    train_perplexity.append(perplexity['train'])
    val_perplexity.append(perplexity['val'])
    save_model(PATH, train_loss_values, val_loss_values,
               train_perplexity, val_perplexity)


print(f'Final train loss: {train_loss_values[-1]:.4f}')
print(f'Final val loss: {val_loss_values[-1]:.4f}')
print(f'Final train perplexity: {train_perplexity[-1]:.4f}')
print(f'Final val perplexity: {val_perplexity[-1]:.4f}')

# Generate text
print(f'Synthesizing text...')
prompt = torch.tensor([[0]], dtype=torch.long, device=device)
if config['sampling'] == "temp":
    sample = decode(model.synthesize(
        prompt, max_new_tokens=config['max_new_tokens'], temperature=config['temperature'])[0], vocab)
elif config['sampling'] == "nucleus":
    sample = decode(model.nucleus_sampling(
        prompt, max_new_tokens=config['max_new_tokens'], theta=config['nucleus'])[0], vocab)
print(sample)
with open(f"{PATH}/text_sample.txt", "w") as file:
    file.write(sample)


spell_checker = SpellChecker()
spelling_accuracy = evaluate_spelling(spell_checker, sample)
with open(f"{PATH}/spelling_accuracy.txt", "w") as file:
    file.write(str(spelling_accuracy))
