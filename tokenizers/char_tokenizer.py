import regex as re
import operator
import numpy as np
import pickle


def process_file(file_path, output_prefix, char_to_idx, idx_to_char):
    with open(file_path, "r") as f:
        text = f.read()

    def encode(s):
        return [char_to_idx[c] for c in s]

    np.save(f"{output_prefix}_char.npy", encode(text))


if __name__ == "__main__":
    char_to_idx = {}
    idx_to_char = {}

    files = [("data/train.txt", "token_data/train"),
             ("data/validation.txt", "token_data/validation"),
             ("data/test.txt", "token_data/test")]

    text = ""
    for file_path, output_prefix in files:
        with open(file_path, "r") as f:
            text = text + f.read()

    for i, ch in enumerate(sorted(list(set(text)))):
        char_to_idx[ch] = i
        idx_to_char[i] = ch

    for file_path, output_prefix in files:
        process_file(file_path, output_prefix, char_to_idx, idx_to_char)

    with open(f'token_data/train_vocabulary_char.pkl', 'wb') as f:
        pickle.dump(idx_to_char, f)
