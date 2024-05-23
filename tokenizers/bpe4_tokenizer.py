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

with open("goblet_book.txt", "r") as f:
   text = f.read()

def get_stats_2D(bytes_list):
    """ 
    Return a dictionary with byte pairs as the keys and the number 
    of occurences as values
    """
    pair_dict = {}
    for word in bytes_list:   
         for pair in zip(word, word[1:]):
            pair_dict[pair] = pair_dict.get(pair, 0) + 1
    return pair_dict

def karpathy_merge_2D(ids, pair, idx):
   """
   Replaces consecutive occurences of pair with the new token idx
   in EVERY word in ids. 
   """
   newids = [] # Two-dimensional list where one inner list is one word.
   for word in ids:
      newids.append(karpathy_merge(word, pair, idx))
   return newids

# From Karpathy
def karpathy_merge(ids, pair, idx):
  """
  Replace all consecutive occurences of pair with the new token idx in ids 
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
    output_bytes = list(map(lambda x : x.decode("utf-8", errors="replace"), output_bytes))
    print(output_bytes)

if __name__ == "__main__":
   # Taken from Karpathy
   GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

   split_pattern = re.compile(GPT4_SPLIT_PATTERN)

   text_split = re.findall(split_pattern, text)

   text_split_utf8 = [list(t.encode("utf-8")) for t in text_split]

   desired_vocabulary_size = 512
   bytes_list, merges = tokenize(bytes_list=text_split_utf8, desired_vocab_size=desired_vocabulary_size)

   vocab = create_vocabulary(merges)

   encoded_text = encode(text, merges)
   np.save("token_data/text_bpe4.npy", encoded_text)

   with open('token_data/vocabulary_bpe4.pkl', 'wb') as f:
      pickle.dump(vocab, f)

#    test_text= "Doom is a 2016 first-person shooter video game developed by id Software and published by Bethesda Softworks. The game is the first major installment in the Doom series since 2004's Doom 3 and was a reboot of the franchise. It was released for PlayStation 4, Windows, and Xbox One in May 2016. A port for Nintendo Switch was co-developed with Panic Button and released in November 2017, and a version for Google Stadia was released in August 2020. Players take the role of an unnamed space marine, known as the Doom Slayer, as he battles demonic forces within an energy-mining facility on Mars and in Hell. Doom was announced as Doom 4 in 2008, and that version underwent an extensive development cycle with different builds and designs before the game was restarted in 2011 and revealed as simply Doom in 2014. It was tested by customers who pre-ordered the 2014 MachineGames game Wolfenstein: The New Order and the general public. Mick Gordon composed the music, with contributions by Richard Devine. The game also has an online multiplayer component and a level editor known as SnapMap, co-developed with Certain Affinity and Escalation Studios respectively.Doom was well received by critics and players. The single-player campaign, graphics, soundtrack, and gameplay received considerable praise, whereas the multiplayer mode drew significant criticism. It was the second best-selling video game in North America and the UK in the week of its release and sold over 500,000 copies for PCs by the end of May 2016. A sequel, Doom Eternal, was released in March 2020."

#    visualize_tokens(encode(test_text, merges), vocab)

#    test_text2 = decode(encode(test_text, merges), vocab)

#    assert test_text == test_text2