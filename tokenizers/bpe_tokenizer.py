import operator
import numpy as np
import pickle

"""
The code in this document is heavily inspired by Andrej Karpathy's 
repo minbpe. Functions copied directly from that repository are 
labeled "From Karpathy". The other functions are rewrites/own implementations
of functions from that repository.
"""

desired_vocabulary_size = 512

def most_common_pair(bytes_list):
    pairs = {}
    for pair in zip(bytes_list, bytes_list[1:]):
        pairs[pair] = pairs.get(pair, 0) + 1
    return max(pairs, key=pairs.get)

# From Karpathy

def get_stats(bytes_list):
    pairs = {}
    for pair in zip(bytes_list, bytes_list[1:]):
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs

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


def tokenize(bytes_list, desired_vocab_size):
    assert desired_vocab_size > 256, "The vocab size cannot be smaller than 256"
    num_merges = desired_vocab_size - 256
    merges = {}
    n = 256
    for _ in range(num_merges):
        pair = most_common_pair(bytes_list)
        merges[pair] = n
        bytes_list = karpathy_merge(bytes_list, pair, n)
        print("merge number", n, pair, "->", "n" )
        n += 1
    return bytes_list, merges


def create_vocabulary(merges):
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for pair, n in merges.items():
        vocab[n] = vocab[pair[0]] + vocab[pair[1]]
    return vocab

# From Karpathy


def decode(ids, vocab):
    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

# From Karpathy


def encode(text, merges):
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = karpathy_merge(tokens, pair, idx)
    return tokens


def process_file(file_path, output_prefix, desired_vocab_size):
    with open(file_path, "r") as f:
        text = f.read()

    bytes_list = list(text.encode("UTF-8"))
    bytes_list, merges = tokenize(bytes_list, desired_vocab_size)
    print("Done with tokenize")
    vocab = create_vocabulary(merges)
    print("Done with creating vocabulary")

    np.save(f"{output_prefix}_bpe.npy", bytes_list)

    with open(f'{output_prefix}_vocabulary_bpe.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Done with saving vocabulary and bytes_list to memory")

    return merges



if __name__ == "__main__":
    print("Starting bpe_tokenizer")
    train_merges = process_file("data/train.txt", "token_data/train", desired_vocab_size=512)
    print("Vocabulary created")
    print("Encoding validation.txt")
    with open("data/validation.txt", "r") as f:
       val_string = f.read()
    val_encoded = encode(val_string, train_merges)
    np.save(f"token_data/validation_bpe.npy", val_encoded)

    print("Encoding test.txt")
    with open("data/test.txt", "r") as f:
       test_string = f.read()
    test_encoded = encode(test_string, train_merges)
    np.save(f"token_data/test_bpe.npy", test_encoded)
