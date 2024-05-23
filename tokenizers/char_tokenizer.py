import regex as re
import operator
import numpy as np
import pickle

char_to_idx = {}
idx_to_char = {}
global_i = 0

def process_file(file_path, output_prefix):
    global char_to_idx
    global idx_to_char
    global global_i
    with open(file_path, "r") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    for ch in chars:
        char_to_idx[ch] = global_i
        idx_to_char[global_i] = ch
        global_i += 1

    def encode(s):
        return [char_to_idx[c] for c in s]

    np.save(f"{output_prefix}_char.npy", encode(text))

if __name__ == "__main__":
    files = [("data/train.txt", "token_data/train"),
             ("data/validation.txt", "token_data/validation"),
             ("data/test.txt", "token_data/test")]

    for file_path, output_prefix in files:
        process_file(file_path, output_prefix)

    # This is not really the train vocabulary. It's the combined vocabularies
    # of train, validation and test
    with open(f'token_data/train_vocabulary_char.pkl', 'wb') as f:
        pickle.dump(idx_to_char, f)
