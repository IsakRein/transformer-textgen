from gensim.models import Word2Vec
import re
import numpy as np
import pickle
import regex as re

# Taken from https://stackoverflow.com/questions/25735644/python-regex-for-splitting-text-into-sentences-sentence-tokenizing
sentence_regex = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"

# Taken from https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists


def flatten(xss):
    return [x for xs in xss for x in xs]


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def process_file(file_path, output_prefix, vector_size=2000):
    with open(file_path, "r") as f:
        raw_text = f.read()

    sentence_split_pattern = re.compile(sentence_regex)
    word_split_pattern = re.compile(GPT4_SPLIT_PATTERN)

    sentences = re.split(sentence_split_pattern, raw_text)

    # Split each sentence into separate words
    for i in range(len(sentences)):
        sentences[i] = re.findall(word_split_pattern, sentences[i])

    model = Word2Vec(sentences=sentences, vector_size=vector_size,
                     min_count=1, workers=4)

    model.save(f"{output_prefix}.model")

    words = flatten(sentences)
    word_set = sorted(list(set(words)))
    vocab = {}
    for i, word in enumerate(word_set):
        vocab[i] = word

    word_vectors = np.zeros((vector_size, len(words)))
    for i, word in enumerate(words):
        word_vectors[:, i] = model.wv[word]

    word_to_index = {}
    for i, word in enumerate(word_set):
        word_to_index[word] = i

    with open(f'{output_prefix}_word_to_index_vec.pkl', 'wb') as f:
        pickle.dump(word_to_index, f)

    with open(f'{output_prefix}_vocabulary_vec.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    np.save(f"{output_prefix}_text_vec.npy", word_vectors)
    np.save(f"{output_prefix}_vec_words.npy", np.array(words))


if __name__ == "__main__":
    files = [("data/train.txt", "token_data/train"),
             ("data/validation.txt", "token_data/validation"),
             ("data/test.txt", "token_data/test")]

    for file_path, output_prefix in files:
        process_file(file_path, output_prefix)
