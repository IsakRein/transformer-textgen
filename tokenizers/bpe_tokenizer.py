import operator
import numpy as np
import pickle

"""
The code in this document is heavily inspired Andrej Karpathy's 
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
    assert desired_vocab_size > 256, "The vocab size cannot be smaller than 256"
    num_merges = desired_vocab_size - 256
    merges = {}
    n = 256
    for _ in range(num_merges):
        pair = most_common_pair(bytes_list)
        merges[pair] = n
        bytes_list = karpathy_merge(bytes_list, pair, n)
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


if __name__ == "__main__":
    with open("goblet_book.txt", "r") as f:
        text = f.read()

    bytes_list = list(text.encode("UTF-8"))
    bytes_list, merges = tokenize(
        bytes_list=bytes_list, desired_vocab_size=desired_vocabulary_size)

    vocab = create_vocabulary(merges)
    test_text = "Doom is a 2016 first-person shooter video game developed by id Software and published by Bethesda Softworks. The game is the first major installment in the Doom series since 2004's Doom 3 and was a reboot of the franchise. It was released for PlayStation 4, Windows, and Xbox One in May 2016. A port for Nintendo Switch was co-developed with Panic Button and released in November 2017, and a version for Google Stadia was released in August 2020. Players take the role of an unnamed space marine, known as the Doom Slayer, as he battles demonic forces within an energy-mining facility on Mars and in Hell. Doom was announced as Doom 4 in 2008, and that version underwent an extensive development cycle with different builds and designs before the game was restarted in 2011 and revealed as simply Doom in 2014. It was tested by customers who pre-ordered the 2014 MachineGames game Wolfenstein: The New Order and the general public. Mick Gordon composed the music, with contributions by Richard Devine. The game also has an online multiplayer component and a level editor known as SnapMap, co-developed with Certain Affinity and Escalation Studios respectively.Doom was well received by critics and players. The single-player campaign, graphics, soundtrack, and gameplay received considerable praise, whereas the multiplayer mode drew significant criticism. It was the second best-selling video game in North America and the UK in the week of its release and sold over 500,000 copies for PCs by the end of May 2016. A sequel, Doom Eternal, was released in March 2020."

    test_text2 = decode(encode(test_text, merges), vocab)
    assert test_text2 == test_text

    np.save("token_data/text_bpe.npy", bytes_list)

    with open('token_data/vocabulary_bpe.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    # with open('vocabulary_bpe.pkl', 'rb') as f:
    #     loaded_vocabulary = pickle.load(f)

    # loaded_bytes_list = np.load("bytes_list.npy", allow_pickle=True)

    # assert np.array_equal(bytes_list, loaded_bytes_list) == True
    # assert loaded_vocabulary == vocab

    # test_text = "Doom is a 2016 first-person shooter video game developed by id Software and published by Bethesda Softworks. The game is the first major installment in the Doom series since 2004's Doom 3 and was a reboot of the franchise. It was released for PlayStation 4, Windows, and Xbox One in May 2016. A port for Nintendo Switch was co-developed with Panic Button and released in November 2017, and a version for Google Stadia was released in August 2020. Players take the role of an unnamed space marine, known as the Doom Slayer, as he battles demonic forces within an energy-mining facility on Mars and in Hell. Doom was announced as Doom 4 in 2008, and that version underwent an extensive development cycle with different builds and designs before the game was restarted in 2011 and revealed as simply Doom in 2014. It was tested by customers who pre-ordered the 2014 MachineGames game Wolfenstein: The New Order and the general public. Mick Gordon composed the music, with contributions by Richard Devine. The game also has an online multiplayer component and a level editor known as SnapMap, co-developed with Certain Affinity and Escalation Studios respectively.Doom was well received by critics and players. The single-player campaign, graphics, soundtrack, and gameplay received considerable praise, whereas the multiplayer mode drew significant criticism. It was the second best-selling video game in North America and the UK in the week of its release and sold over 500,000 copies for PCs by the end of May 2016. A sequel, Doom Eternal, was released in March 2020."

    # test_text2 = decode(encode(test_text, merges), vocab)
    # assert test_text2 == test_text
