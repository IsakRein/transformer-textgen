import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import time
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def load_data_from_file(file):
    with open(file, 'r') as f:
        data = f.read()
    book_data = np.array(list(data))
    book_chars = np.unique(book_data)
    char_to_ind = {char: idx for idx, char in enumerate(book_chars)}
    ind_to_char = {idx: char for idx,
                   char in enumerate(book_chars)}

    encoded = torch.tensor([char_to_ind[ch]
                           for ch in book_data], device=device, dtype=torch.long)
    K = len(book_chars)
    return book_chars, char_to_ind, ind_to_char, encoded, K


def get_seq(data, seq_length, batch_size, idx, char_to_ind):
    X = torch.zeros((batch_size, seq_length),
                    dtype=int, device=device)
    Y = torch.zeros((batch_size, seq_length), dtype=int, device=device)

    for batch in range(batch_size):
        X_batch = data[idx:idx+seq_length]
        Y_batch = data[idx+1:idx+seq_length+1]

        X[batch, :] = X_batch
        Y[batch, :] = Y_batch

    X.to(device)
    Y.to(device)

    return X, Y


def one_hot_encode(data, K):
    one_hot = torch.zeros((data.shape[0], data.shape[1], K), device=device)
    one_hot = one_hot.scatter(2, data.unsqueeze(-1), 1)
    return one_hot


def data_to_text(data, ind_to_char):
    return ''.join([ind_to_char[ind.item()] for ind in data])


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, batch_size: int = 1):
        super().__init__()
        self.d_model = d_model  # Dimension of vectors (512)
        self.vocab_size = vocab_size  # Size of the vocabulary
        # PyTorch layer that converts integer indices to dense embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Normalizing the variance of the embeddings
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(
            0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    # TODO: REWRITE. Taken from: https://ai.plainenglish.io/building-and-training-a-transformer-from-scratch-fdbf3db00df4

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):

        d_k = query.shape[-1]  # The last dimension of query, key, and value

        # We calculate the Attention(Q,K,V) as in the formula in the image above
        # @ = Matrix multiplication sign in PyTorch
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Before applying the softmax, we apply the mask to hide some interactions between words
        if mask is not None:  # If a mask IS defined...
            # Replace each value where mask is equal to 0 by -1e9
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # Applying softmax
        if dropout is not None:  # If a dropout IS defined...
            # We apply dropout to prevent overfitting
            attention_scores = dropout(attention_scores)

        # Multiply the output matrix by the V matrix, as in the formula
        return (attention_scores @ value)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = self.W_q(q)  # Q' matrix
        key = self.W_k(k)  # K' matrix
        value = self.W_v(v)  # V' matrix

        batch_size = query.shape[0]
        sequence_length = query.shape[1]

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2)  # Transpose => bring the head to the second dimension
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(
            1, 2)  # Transpose => bring the head to the second dimension
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2)  # Transpose => bring the head to the second dimension

        attention = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attention = attention.masked_fill(
                mask == False, torch.tensor(float('-inf')))

        x = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(
            x.shape[0], -1, self.h * self.d_k)

        return self.W_o(x)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=True)
        self.W2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.W1(x)
        res = F.relu(res)
        res = self.dropout(res)
        res = self.W2(res)

        return res


class LayerNorm(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)
        res = res + x
        return res


class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout) -> None:
        super().__init__()
        self.multiHeadAttentionBlock = MultiHeadAttentionBlock(
            d_model, h, dropout)
        self.residualConnection1 = ResidualConnection(dropout)
        self.layerNorm1 = LayerNorm()
        self.mlp = MLP(d_model, d_ff, dropout)
        self.residualConnection2 = ResidualConnection(dropout)
        self.layerNorm2 = LayerNorm()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # TODO: Fixa så att layerNorm är här istället för i ResidualConnection
        x = self.residualConnection1(
            x, lambda x: self.multiHeadAttentionBlock(x, x, x, mask))
        # x = self.layerNorm1(x)
        x = self.residualConnection2(x, lambda x: self.mlp(x))
        # x = self.layerNorm2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, h, d_ff, n_layers, dropout) -> None:
        super().__init__()
        self.encoderBlocks = nn.ModuleList(
            [EncoderBlock(d_model, h, d_ff, dropout) for _ in range(n_layers)])
        self.norm = LayerNorm()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for encoderBlock in self.encoderBlocks:
            x = encoderBlock(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout) -> None:
        super().__init__()
        self.selfAttentionBlock = MultiHeadAttentionBlock(
            d_model, h, dropout)
        self.residualConnection1 = ResidualConnection(dropout)
        self.crossAttentionBlock = MultiHeadAttentionBlock(
            d_model, h, dropout)
        self.residualConnection2 = ResidualConnection(dropout)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.residualConnection3 = ResidualConnection(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        res = self.residualConnection1(
            x, lambda x: self.selfAttentionBlock(x, x, x, tgt_mask))
        res = self.residualConnection2(
            res, lambda res: self.crossAttentionBlock(res, encoder_output, encoder_output, src_mask))
        res = self.residualConnection3(res, lambda res: self.mlp(res))

        return res


class Decoder(nn.Module):
    def __init__(self, d_model, h, d_ff, n_layers, dropout) -> None:
        super().__init__()
        self.decoderBlocks = nn.ModuleList(
            [DecoderBlock(d_model, h, d_ff, dropout) for _ in range(n_layers)])
        self.norm = LayerNorm()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for decoderBlock in self.decoderBlocks:
            x = decoderBlock(x, encoder_output, src_mask, tgt_mask)
        x = self.norm(x)
        projection = torch.log_softmax(self.projection(x), dim=-1)
        return projection


class Transformer(nn.Module):
    def __init__(self, d_model, h, d_ff, n_layers, dropout) -> None:
        super().__init__()
        self.encoder = Encoder(d_model, h, d_ff, n_layers, dropout)
        self.decoder = Decoder(d_model, h, d_ff, n_layers, dropout)
        self.inputEmbedding = InputEmbedding(d_model, vocab_size)
        self.positionalEncoding = PositionalEncoding(d_model, seq_length)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, X, Y):
        X_mask, Y_mask = self.generate_mask(X, Y)

        X = self.positionalEncoding(self.inputEmbedding(X))
        Y = self.positionalEncoding(self.inputEmbedding(Y))

        output = self.decoder(Y, self.encoder(X, X_mask), X_mask, Y_mask)
        return output


torch.manual_seed(0)
book_chars, char_to_ind, ind_to_char, encoded_data, K = load_data_from_file(
    './sample_data/goblet_book.txt')


d_model = 512
nhead = 8
vocab_size = K
seq_length = 25
batch_size = 1
d_ff = 2048
n_layers = 6
dropout = 0.1

model = Transformer(d_model, nhead, d_ff, n_layers, dropout)
model.to(device)

criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
iteration = 0


for epoch in range(1, 10):
    print(f'-------------')
    print(f'Epoch {epoch}')

    model.zero_grad()

    for i in range(0, len(encoded_data) - seq_length, seq_length):
        # Prepare inputs and targets
        X, Y = get_seq(encoded_data, seq_length, batch_size, i, char_to_ind)
        model.zero_grad()
        output = model(X, Y)

        unbatched_output = torch.reshape(
            output, (batch_size * seq_length, K))
        unbatched_output_Y = torch.reshape(one_hot_encode(
            Y, vocab_size), (batch_size * seq_length, K))
        # Backward pass
        loss = criterion(unbatched_output, unbatched_output_Y)
        loss = torch.sum(loss) / batch_size

        loss.backward()
        optimizer.step()

        # print("------")
        # print(output)
        predicted_text = data_to_text(torch.argmax(
            output.squeeze(0), axis=1), ind_to_char)
        print(f'Predicted text:\n {predicted_text}')

        actual_text = data_to_text(Y.squeeze(0), ind_to_char)
        print(f'Actual text:\n {actual_text}')
        print("------")

        if iteration % 10 == 0:
            print(f'Iteration: {iteration}, Loss: {loss.item()}')

        iteration += 1
