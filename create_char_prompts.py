import pickle


def load_vocab(tokenizer):
    with open(f"token_data/train_vocabulary_{tokenizer}.pkl", "rb") as f:
        vocab = pickle.load(f)

    merges = {}
    for i, token in vocab.items():
        merges[token] = i

    return vocab, merges


def encode(text, merges):
    tokens = [merges[c] for c in text]
    return tokens


if __name__ == "__main__":
    vocab, merges = load_vocab("char")
    txt = "On 4 May 2008 , Torres scored"
    print(encode(txt, merges))
