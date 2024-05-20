import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
import pickle


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
        len(data) - config['seq_length'], (config['batch_size'],))
    X_batch = torch.stack([data[j:j+config['seq_length']] for j in idx])
    Y_batch = torch.stack([data[j+1:j+config['seq_length']+1] for j in idx])
    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
    return X_batch, Y_batch


def load_bpe_gpt4():
    """
    Loads the GPT4-tokenizer
    """
    # byte_list is the entire text encoded into tokens. Se more in bpe.py
    byte_list = torch.tensor(
        np.load("encoded_text_gpt4.npy"), dtype=torch.long)

    # token_vocabulary is used to decode tokens to chars
    with open("vocabulary_gpt4.pkl", "rb") as f:
        token_vocabulary = pickle.load(f)
    return byte_list, token_vocabulary

# From Karpathy
def decode(ids, vocab):
    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx.item()] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        for k in range(config['eval_iters']):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


class PositionalEncoding(nn.Module):
    def __init__(self, n_embd, seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(seq_length, n_embd)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float()
                             * -(math.log(10000.0) / n_embd))

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
        self.register_buffer('tril', torch.tril(
            torch.ones(config['seq_length'], config['seq_length'])))

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
    def __init__(self, ) -> None:
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
        self.embed_tokens = nn.Embedding(
            vocab_size, config['n_embd'])
        self.positional_encoding = PositionalEncoding(
            config['n_embd'], config['seq_length'])
        self.decoder_blocks = nn.Sequential(
            *[DecoderBlock() for _ in range(config['n_layer'])])
        self.norm = nn.LayerNorm(config['n_embd'])
        self.transform_to_vocab_size = nn.Linear(
            config['n_embd'], vocab_size)


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


# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Set random seed and device
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# text_processor = TextProcessor(text)
# data = torch.tensor(text_processor.encode(text), dtype=torch.long)

data, vocab = load_bpe_gpt4()

n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

# Initialize model and optimizer
model = DecoderOnlyTransformer(len(vocab)).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config['learning_rate'], weight_decay=config['lambda'])

torch.save(model.state_dict(), 'model.pth')
# Training loop
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

for iter in range(config['max_iters']):
    X_batch, Y_batch = get_batch('train')
    logits, loss = model(X_batch, Y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % config['eval_interval'] == 0 or iter == config['max_iters'] - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {
              losses['train']:.4f}, val loss {losses['val']:.4f}")

    if iter % 1000 == 0:
        prompt = torch.tensor([[0]], dtype=torch.long, device=device)
        print(decode(model.synthesize(
            prompt, max_new_tokens=config['max_new_tokens'], temperature=config['temperature'])[0], vocab))


losses = estimate_loss()
print(f'Final train loss: {losses["train"]:.4f}')
print(f'Final val loss: {losses["val"]:.4f}')

# Generate text
print(f'Synthesizing text...')

prompt = torch.tensor([[0]], dtype=torch.long, device=device)
print(decode(model.synthesize(
    prompt, max_new_tokens=config['max_new_tokens'], temperature=config['temperature'])[0], vocab))
