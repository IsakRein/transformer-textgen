import operator
import numpy as np
import pickle 

desired_vocabulary_size = 1000

# From Karpathy
def get_stats(bytes_list):
    """ 
    Return a dictionary with byte pairs as the keys and the number 
    of occurences as values
    """
    pair_dict = {}
    for pair in zip(bytes_list, bytes_list[1:]):
        pair_dict[pair] = pair_dict.get(pair, 0) + 1
    return pair_dict

def replace_byte_sequence(byte_list, to_replace, replacement):
    i = 0
    byte_list_len = len(byte_list)
    num_removal = 0
    while i < len(bytes_list) - 1:
        seq = (byte_list[i], byte_list[i+1])
        if seq == to_replace:
            del bytes_list[i]
            bytes_list[i] = replacement 
            num_removal += 1
        i += 1
    assert byte_list_len - len(byte_list) == num_removal
    return bytes_list

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

def tokenize(bytes_list, desired_vocab_size):
    i = 0
    n = 256
    num_replacements = desired_vocab_size - n
    merges = {}
    while i < num_replacements:
       pair_dict = get_stats(bytes_list=bytes_list)
       most_common_pair = max(pair_dict.items(), key=operator.itemgetter(1))[0]
       bytes_list = karpathy_merge(bytes_list, most_common_pair, n)
       merges[most_common_pair] = n
       print("merge:", most_common_pair, ":", merges[most_common_pair]) 
       n += 1
       i += 1
    return bytes_list, merges

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
            break # nothing else can be merged
        idx = merges[pair]
        tokens = karpathy_merge(tokens, pair, idx)
    return tokens

with open("goblet_book.txt", "r") as f:
   text = f.read()

bytes_list = list(text.encode("UTF-8"))
bytes_list, merges  = tokenize(bytes_list=bytes_list, desired_vocab_size=desired_vocabulary_size)

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

np.save("bytes_list.npy", bytes_list)

with open('vocabulary.pkl', 'wb') as f:
    pickle.dump(vocab, f)

with open('vocabulary.pkl', 'rb') as f:
    loaded_vocabulary = pickle.load(f)

loaded_bytes_list = np.load("bytes_list.npy", allow_pickle=True)

assert np.array_equal(bytes_list, loaded_bytes_list) == True
assert loaded_vocabulary == vocab

test_text= "Doom is a 2016 first-person shooter video game developed by id Software and published by Bethesda Softworks. The game is the first major installment in the Doom series since 2004's Doom 3 and was a reboot of the franchise. It was released for PlayStation 4, Windows, and Xbox One in May 2016. A port for Nintendo Switch was co-developed with Panic Button and released in November 2017, and a version for Google Stadia was released in August 2020. Players take the role of an unnamed space marine, known as the Doom Slayer, as he battles demonic forces within an energy-mining facility on Mars and in Hell. Doom was announced as Doom 4 in 2008, and that version underwent an extensive development cycle with different builds and designs before the game was restarted in 2011 and revealed as simply Doom in 2014. It was tested by customers who pre-ordered the 2014 MachineGames game Wolfenstein: The New Order and the general public. Mick Gordon composed the music, with contributions by Richard Devine. The game also has an online multiplayer component and a level editor known as SnapMap, co-developed with Certain Affinity and Escalation Studios respectively.Doom was well received by critics and players. The single-player campaign, graphics, soundtrack, and gameplay received considerable praise, whereas the multiplayer mode drew significant criticism. It was the second best-selling video game in North America and the UK in the week of its release and sold over 500,000 copies for PCs by the end of May 2016. A sequel, Doom Eternal, was released in March 2020."

test_text2 = decode(encode(test_text, merges), vocab)
assert test_text2 == test_text