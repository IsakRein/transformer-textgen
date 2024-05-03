from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import re 

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

sentences = process_raw_text_file("goblet_book.txt")
print(sentences)
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

model = Word2Vec.load("word2vec.model")

print(model.wv.most_similar(positive=["scar", "boy"], negative=[]))

