import regex as re
import operator
import numpy as np
import pickle

with open("goblet_book.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [char_to_idx[c] for c in s]


np.save("token_data/text_char.npy", encode(text))
with open('token_data/vocabulary_char.pkl', 'wb') as f:
    pickle.dump(idx_to_char, f)
