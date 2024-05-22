import regex as re
import operator
from bpe_tokenizer import encode, decode
import numpy as np
import pickle

"""
The code in this document is heavily inspired Andrej Karpathy's 
repo minbpe. Functions copied directly from that repository are 
labeled "From Karpathy". The other functions are rewrites/own implementations
of functions from that repository.
"""


def get_stats_2D(bytes_list):
    """ 
    Return a dictionary with byte pairs as the keys and the number 
    of occurrences as values
    """
    pair_dict = {}
    for word in bytes_list:
        for pair in zip(word, word[1:]):
            pair_dict[pair] = pair_dict.get(pair, 0) + 1
    return pair_dict


def karpathy_merge_2D(ids, pair, idx):
   """
   Replaces consecutive occurrences of pair with the new token idx
   in EVERY word in ids. 
   """
   newids = []  # Two-dimensional list where one inner list is one word.
   for word in ids:
      newids.append(karpathy_merge(word, pair, idx))
   return newids

# From Karpathy


def karpathy_merge(ids, pair, idx):
  """
  Replace all consecutive occurrences of pair with the new token idx in ids 
  """
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids


def create_vocabulary(merges):
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for pair, n in merges.items():
        vocab[n] = vocab[pair[0]] + vocab[pair[1]]
    return vocab


def tokenize(bytes_list, desired_vocab_size):
    assert desired_vocab_size > 256, "The vocab size cannot be smaller than 256"
    num_merges = desired_vocab_size - 256
    merges = {}
    n = 256
    for _ in range(num_merges):
        pairs = get_stats_2D(bytes_list)
        pair = max(pairs, key=pairs.get)
        merges[pair] = n
        bytes_list = karpathy_merge_2D(bytes_list, pair, n)
        n += 1
    return bytes_list, merges


def visualize_tokens(token_indices, vocab):
    output_bytes = [vocab[idx] for idx in token_indices]
    output_bytes = list(map(lambda x: x.decode(
        "utf-8", errors="replace"), output_bytes))
    print(output_bytes)


def process_file(file_path, output_prefix, desired_vocabulary_size=500):
    with open(file_path, "r") as f:
        text = f.read()

    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    split_pattern = re.compile(GPT4_SPLIT_PATTERN)

    text_split = re.findall(split_pattern, text)
    text_split_utf8 = [list(t.encode("utf-8")) for t in text_split]

    bytes_list, merges = tokenize(
        bytes_list=text_split_utf8, desired_vocab_size=desired_vocabulary_size)
    vocab = create_vocabulary(merges)

    encoded_text = encode(text, merges)
    np.save(f"{output_prefix}_bpe.npy", encoded_text)

    with open(f'{output_prefix}_vocabulary_bpe.pkl', 'wb') as f:
        pickle.dump(vocab, f)


if __name__ == "__main__":
    files = [("data/train.txt", "token_data/train"),
             ("data/validation.txt", "token_data/validation"),
             ("data/test.txt", "token_data/test")]

    for file_path, output_prefix in files:
        process_file(file_path, output_prefix)
