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

# Taken from https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
def flatten(xss):
    return [x for xs in xss for x in xs]

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
   #newids = []  # Two-dimensional list where one inner list is one word.
   newids = [[] for _ in range(len(ids))]
   for i, word in enumerate(ids):
      #newids.append(karpathy_merge(word, pair, idx))
      newids[i] = karpathy_merge(word, pair, idx)
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
        print("merge number", n, pair, "->", n )
        n += 1
    return bytes_list, merges


def visualize_tokens(token_indices, vocab):
    output_bytes = [vocab[idx] for idx in token_indices]
    output_bytes = list(map(lambda x: x.decode(
        "utf-8", errors="replace"), output_bytes))
    print(output_bytes)

def process_file(file_path, output_prefix, desired_vocabulary_size):
    with open(file_path, "r") as f:
        text = f.read()

    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    split_pattern = re.compile(GPT4_SPLIT_PATTERN)

    text_split = re.findall(split_pattern, text)
    text_split_utf8 = [list(t.encode("utf-8")) for t in text_split]

    bytes_list, merges = tokenize(bytes_list=text_split_utf8, desired_vocab_size=desired_vocabulary_size)
    print("Done with tokenize")
    bytes_list = flatten(bytes_list)
    bytes_list = np.asanyarray(bytes_list)
    vocab = create_vocabulary(merges)
    print("Done with create_vocabulary")

    np.save(f"{output_prefix}_bpe4.npy", bytes_list)

    with open(f'{output_prefix}_vocabulary_bpe4.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Done with saving vocabulary and bytes_list to memory")
    # Test that tokenize works correctly
    encoded_text = encode(text, merges)
    assert np.array_equal(encoded_text, bytes_list)
    
    return merges
    

if __name__ == "__main__":
    print("Starting bpe4_tokenizer")
    train_merges = process_file("data/train.txt", "token_data/train", desired_vocabulary_size=512)
    print("Vocabulary created")
    print("Encoding validation.txt")
    with open("data/validation.txt", "r") as f:
       val_string = f.read()
    val_encoded = encode(val_string, train_merges)
    np.save(f"token_data/validation_bpe4.npy", val_encoded)

    print("Encoding test.txt")
    with open("data/test.txt", "r") as f:
       test_string = f.read()
    test_encoded = encode(test_string, train_merges)
    np.save(f"token_data/test_bpe4.npy", test_encoded)
