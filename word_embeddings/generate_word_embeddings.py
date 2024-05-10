from gensim.test.utils import common_texts
from gensim.models import Word2Vec, KeyedVectors
import gensim
import gensim.downloader as api
import re 
import numpy as np
import pickle

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

vector_size = 80

sentences = process_raw_text_file("goblet_book.txt")
model = Word2Vec(sentences=sentences, vector_size=vector_size, min_count=1, workers=4)

#model.save("word2vec_80.model")

words = flatten(sentences)
word_set = sorted(list(set(words)))
word_vectors = np.zeros((vector_size, len(words)))

for i, word in enumerate(words):
    word_vectors[:, i] = model.wv[word]

word_to_index = {}
for i, word in enumerate(word_set):
    word_to_index[word] = i

with open('word_to_index.pkl', 'wb') as f:
    pickle.dump(word_to_index, f)

#np.save("word_vectors_80.npy", word_vectors)
#np.save("words_80.npy", np.array(words))
np.save("word_set", np.array(word_set))



