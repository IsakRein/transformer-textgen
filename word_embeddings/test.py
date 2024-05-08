from gensim.test.utils import common_texts
from gensim.models import Word2Vec, KeyedVectors
import gensim
import gensim.downloader as api
import re 
import numpy as np

def process_raw_text_file(path):
    def remove_punctuation(lst):
        for i in range(len(lst)):
            lst[i] = re.sub(r"[\.,\?\!\"]+", "", lst[i]).lower()
        return lst
    with open(path, "r") as f:
        raw_text = f.read()
        sentence_split = raw_text.split(".")
        remove_empty_str = lambda x: False if x == '' else True 
        remove_empty_arr = lambda x: False if x == [] else True 

        non_empty_sentences = list(filter(remove_empty_str, sentence_split))
        non_empty_word_sentences = [re.split(r"\s+", s) for s in non_empty_sentences]
        non_empty_word_sentences = [remove_punctuation(s) for s in non_empty_word_sentences]
        non_empty_word_sentences = [list(filter(remove_empty_str, l)) for l in non_empty_word_sentences]
        non_empty_word_sentences = list(filter(remove_empty_arr, non_empty_word_sentences))

        return non_empty_word_sentences

# Taken from https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
def flatten(xss):
    return [x for xs in xss for x in xs]

vector_size = 2000

sentences = process_raw_text_file("goblet_book.txt")
model = Word2Vec(sentences=sentences, vector_size=vector_size, min_count=1, workers=4)

model.save("word2vec.model")

words = flatten(sentences)

word_vectors = np.zeros((vector_size, len(words)))
for i, word in enumerate(words):
    print("word", word)
    word_vectors[:, i] = model.wv[word]

np.save("word_vectors.npy", word_vectors)
np.save("words.npy", np.array(words))


