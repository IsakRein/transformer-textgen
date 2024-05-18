import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class TextProcessor:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def decode_tensor(self, t):
        return ''.join([self.itos[i.item()] for i in t])


def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(
        len(data) - config['block_size'], (config['batch_size'],))
    X_batch = torch.stack([data[j:j+config['block_size']] for j in idx])
    Y_batch = torch.stack([data[j+1:j+config['block_size']+1] for j in idx])
    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
    return X_batch, Y_batch


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


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config['n_embd'], head_size, bias=False)
        self.query = nn.Linear(config['n_embd'], head_size, bias=False)
        self.value = nn.Linear(config['n_embd'], head_size, bias=False)
        self.dropout = nn.Dropout(config['dropout'])
        self.register_buffer('tril', torch.tril(
            torch.ones(config['block_size'], config['block_size'])))

    def forward(self, X):
        B, T, C = X.shape
        k, q, v = self.key(X), self.query(X), self.value(X)
        affinity = q @ k.transpose(-2, -1) / (self.key.weight.size(-1) ** 0.5)
        affinity = affinity.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
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


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * config['feed_forward_multiplier']),
            nn.ReLU(),
            nn.Linear(n_embd * config['feed_forward_multiplier'], n_embd),
            nn.Dropout(config['dropout'])
        )

    def forward(self, X):
        return self.net(X)


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = config['n_embd'] // config['n_head']
        self.heads = MultiheadAttention(config['n_head'], head_size)
        self.mlp = FeedForward(config['n_embd'])
        self.norm1 = nn.LayerNorm(config['n_embd'])
        self.norm2 = nn.LayerNorm(config['n_embd'])

    def forward(self, X):
        # TODO: This is a simple way to do residual connections.
        # Maybe replace with our own for clarity.
        X = X + self.heads(self.norm1(X))
        X = X + self.mlp(self.norm2(X))
        return X


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            text_processor.vocab_size, config['n_embd'])
        self.position_embedding_table = nn.Embedding(
            config['block_size'], config['n_embd'])
        self.decoder_blocks = nn.Sequential(
            *[DecoderBlock() for _ in range(config['n_layer'])])
        self.norm = nn.LayerNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], text_processor.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, X, Y=None):
        B, T = X.shape
        # TODO replace with fancier token / position embedding
        token_embeddings = self.token_embedding_table(X)
        position_embeddings = self.position_embedding_table(
            torch.arange(T, device=device))
        X = token_embeddings + position_embeddings
        X = self.decoder_blocks(X)
        X = self.norm(X)
        logits = self.lm_head(X)
        if Y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            Y = Y.view(B*T)
            loss = F.cross_entropy(logits, Y)
        return logits, loss

    # TODO: Maybe change. Taken from Andrej Karpathy's blog
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Set random seed and device
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read input text
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text_processor = TextProcessor(text)
data = torch.tensor(text_processor.encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

# Initialize model and optimizer
model = Transformer().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config['learning_rate'])

# Training loop
print(f"Model has {sum(p.numel()
                       for p in model.parameters()):,} parameters")

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

losses = estimate_loss()
print(f'Final train loss: {losses['train']:.4f}')
print(f'Final val loss: {losses['val']:.4f}')

# Generate text
print(f'Synthesizing text...')
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(text_processor.decode_tensor(model.generate(
    context, max_new_tokens=config['max_new_tokens'])[0]))
