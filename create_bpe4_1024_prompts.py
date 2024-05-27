import pickle


def load_vocab(tokenizer):
    with open(f"token_data/train_vocabulary_{tokenizer}.pkl", "rb") as f:
        vocab = pickle.load(f)
    return vocab


def load_data(tokenizer):
    with open(f"token_data/train_vocabulary_{tokenizer}.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(f"token_data/train_merges_{tokenizer}.pkl", "rb") as f:
        merges = pickle.load(f)
    return vocab, merges


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


def decode(ids, vocab):
    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


if __name__ == "__main__":
    vocab, merges = load_data("bpe4_1024")
    txt = "On 4 May 2008 , Torres scored"
    print(encode(txt, merges))
