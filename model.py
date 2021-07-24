from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, LSTM

import numpy as np
import random

from gensim.models import KeyedVectors
import gensim.downloader as api

import dataset

path = api.load('glove-twitter-25', True)
vecs = KeyedVectors.load_word2vec_format(path)

BATCH_SIZE = 8

MEAN_LEN = 7    # words
MAX_LEN = 70    # Max length of a sentence in words

class Punctuator:
    def __init__(self):
        self.model = self.make_model()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model.summary()

    @staticmethod
    def make_model():
        return keras.models.Sequential([
            get_keras_embedding(vecs),     # Embedding layer. int -> 25 float  $ https://stackoverflow.com/questions/51492778/how-to-properly-use-get-keras-embedding-in-gensim-s-word2vec
            Dense(20, input_shape=(MAX_LEN, 25)),
            LSTM(20, return_sequences=True),    # We want to see the output for each letter, not just the last.
            Dense(*dataset.OUTPUT_SHAPE, activation='softmax'),
        ])

    def train(self, gen, valgen=None):
        self.model.fit(gen, steps_per_epoch=21000//BATCH_SIZE, epochs=10)

    @staticmethod
    def generator(x, y):
        assert len(x) == len(y)

        # We always want a training sample to start at the beginning of a sentence. Find the indices of the words that start a sentence.
        starts = [i+1 for i in range(len(y) - 2*MAX_LEN) if np.argmax(y[i]) != 0]   # Get the indexes one above those where there's punctuation.


        while True:
            xs = np.zeros((BATCH_SIZE, MAX_LEN), dtype=np.int64)
            ys = np.zeros((BATCH_SIZE, MAX_LEN, 4))

            for i in range(BATCH_SIZE):
                idx = random.choice(starts)

                xs[i] = Punctuator.encode(x[idx:idx+MAX_LEN])
                ys[i] = y[idx:idx+MAX_LEN]

            yield xs, ys

    @staticmethod
    def encode(text: [str]) -> [int]:
        placeholder = vecs.vocab['hmm']   # should be pretty neutral
        return [vecs.vocab.get(word, placeholder).index for word in text]

# Code comes from [here]. We just had to copy it so that it uses the tf.keras and not keras. https://github.com/RaRe-Technologies/gensim/blob/a811a231747ba5b089d74c4bc22e8f419874baa1/gensim/models/keyedvectors.py#L1386
def get_keras_embedding(v, train_embeddings=False, word_index=None):
    weights = v.vectors
    layer = Embedding(
        input_dim=weights.shape[0], output_dim=weights.shape[1],
        weights=[weights], trainable=train_embeddings
    )
    return layer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true')
    opts=parser.parse_args()

    p = Punctuator()
    x, y = dataset.make_data(dataset.load_sentences('train'))
    x_val, y_val = dataset.make_data(dataset.load_sentences('test'))

    try:
        p.train(Punctuator.generator(x, y), valgen=Punctuator.generator(x, y))
    finally:
        p.model.save('punctuator.h5')
