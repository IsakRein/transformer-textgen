import regex as re
import operator
import numpy as np
import pickle


def process_file(file_path, output_prefix):
    with open(file_path, "r") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [char_to_idx[c] for c in s]

    np.save(f"{output_prefix}_char.npy", encode(text))
    with open(f'{output_prefix}_vocabulary_char.pkl', 'wb') as f:
        pickle.dump(idx_to_char, f)


if __name__ == "__main__":
    files = [("data/train.txt", "token_data/train"),
             ("data/validation.txt", "token_data/validation"),
             ("data/test.txt", "token_data/test")]

    for file_path, output_prefix in files:
        process_file(file_path, output_prefix)
