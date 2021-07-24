from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.callbacks import LambdaCallback

import numpy as np
import random

from gensim.models import KeyedVectors
import gensim.downloader as api

import dataset
import re

path = api.load('glove-twitter-50', True)
vecs = KeyedVectors.load_word2vec_format(path)

BATCH_SIZE = 32

MEAN_LEN = 7    # words
MAX_LEN = 70    # Max length of a sentence in words

# 50: .91 val

class Punctuator:
    def __init__(self):
        self.model, self.trainable_model = self.make_model()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.trainable_model.summary()

    @staticmethod
    def make_model():
        trainable_model = keras.models.Sequential([
            Dense(50, activation='relu', input_shape=(MAX_LEN, 50)),
            Dense(30),
            Bidirectional(LSTM(20, activation='tanh', return_sequences=True), merge_mode='mul'),    # We want to see the output for each letter, not just the last.
            Dense(10, activation='relu'),
            Dense(*dataset.OUTPUT_SHAPE, activation='softmax'),
        ])

        full_model = keras.models.Sequential([
            get_keras_embedding(vecs),     # Embedding layer. int -> 25 float  $ https://stackoverflow.com/questions/51492778/how-to-properly-use-get-keras-embedding-in-gensim-s-word2vec
            trainable_model,
        ])

        return full_model, trainable_model

    def train(self, gen, valgen=None):
        self.model.fit(
            gen,
            steps_per_epoch=21000//BATCH_SIZE,
            epochs=100,
            validation_data=valgen,
            validation_steps=1,         # else it would loop infinitely on the val generator
            callbacks=[LambdaCallback(
                on_batch_start=lambda *args: self.on_epoch_start(*args, valgen=valgen),
            )]
        )

    def on_epoch_start(self, epoch, logs, valgen=None):
        test_words = [vecs.index2word[idx] for idx in next(valgen)[0][0]]   # The generator already tokenizes it, so we have to untokenize it again.
        test_prediction = self.predict(test_words)
        print({
        "loss": logs['loss'],
        "val_loss": logs['val_loss'],
        "val_acc": logs['val_acc'],
        "test_words": ' '.join(test_words),
        "test_prediction": test_prediction,
        })

        wandb.log({
            "loss": logs['loss'],
            "val_loss": logs['val_loss'],
            "val_acc": logs['val_acc'],
            "test_words": ' '.join(test_words),
            "test_prediction": test_prediction,
        })

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
        return [vecs.vocab.get(word, placeholder).index for word in text]   # https://stackoverflow.com/questions/51492778/how-to-properly-use-get-keras-embedding-in-gensim-s-word2vec

    def predict(self, words: [str]):
        if len(words) > MAX_LEN:
            raise Exception(f'Length of string to punctuate must be less than {MAX_LEN} words.')

        encoded = np.array([self.encode(words)])
        probabilities = self.model.predict(encoded)[0]  # shape: (1, MAX_LEN, 4)

        output = ''
        for i in range(len(words)):
            word = words[i]
            punct = dataset.OUTPUT_MAP[np.argmax(probabilities[i])]

            if np.argmax(probabilities[i]) != 0:    # If the previous word had a terminator after it
                word = word.title()

            output += punct + ' ' + word

        return output

# Code comes from [here]. We just had to copy it so that it uses the tf.keras and not keras. https://github.com/RaRe-Technologies/gensim/blob/a811a231747ba5b089d74c4bc22e8f419874baa1/gensim/models/keyedvectors.py#L1386
def get_keras_embedding(v, train_embeddings=False, word_index=None):
    weights = v.vectors
    layer = Embedding(
        input_dim=weights.shape[0], output_dim=weights.shape[1],
        weights=[weights], trainable=train_embeddings
    )
    return layer

if __name__ == '__main__':
    import wandb
    wandb.init('sentence-termination')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--test', action='store_true')
    opts=parser.parse_args()

    p = Punctuator()

    if opts.load:
        p.trainable_model.load_weights('punctuator.h5')

    x, y = dataset.make_data(dataset.load_sentences('train'))
    x_val, y_val = dataset.make_data(dataset.load_sentences('test'))

    if opts.test:
        while True:
            text = input(' > ').lower()
            result = p.predict(re.sub('(?!\w| ).', '', text).split())
            print(f'-> {result}\n')
    else:
        try:
            p.train(Punctuator.generator(x, y), valgen=Punctuator.generator(x_val, y_val))
        finally:
            p.trainable_model.save('punctuator.h5')
