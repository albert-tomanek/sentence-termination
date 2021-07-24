import glob
import re

from srtfile import subtitle_reader

def load_sentences(dirpath) -> str:
    srt_texts = [' '.join(sub.text for sub in subtitle_reader(file)) for file in glob.glob(dirpath + '/*.srt')]
    corpus = ' '.join(srt_texts)

    return corpus.replace('\n', ' ')

SENTENCE_END = '\.\.\.|…|\.|\!|\?'

# OUTPUTS = {
#     '':    [1, 0, 0, 0],    # For when there's no punctuation after the word
#     '.':   [0, 1, 0, 0],
#     '...': [0, 1, 0, 0],
#     '…':   [0, 1, 0, 0],
#     '!':   [0, 0, 1, 0],
#     '?':   [0, 0, 0, 1],
# }
OUTPUTS = {
    '':    [1, 0, 0, 0],    # For when there's no punctuation after the word
    '.':   [0, 1, 0, 0],
    '...': [0, 1, 0, 0],
    '…':   [0, 1, 0, 0],
    '!':   [0, 1, 0, 0],
    '?':   [0, 1, 0, 0],
}
OUTPUT_SHAPE = (4,)
OUTPUT_MAP = ['', '.', '!', '?']    # This is used to convert the indices from the above table back into punctuaiton.

def make_data(text):
    text = re.sub('(?!\w| |'+SENTENCE_END+').', '', text.lower())    # Get rid of commas, they're too nuanced for prediction
    text = re.sub('(?=\w)\d+(?=\w)', 'several', text)   # Use the same embedding for all numbers. The word 'several' can be used in the same grammatic context as a number.

    x: [str] = []
    y: [(float, float, float, float)] = []  # The probability that each punctuation comes after this word

    matches = list(re.finditer(SENTENCE_END, text))
    for i in range(len(matches)):
        start = matches[i-1].end() if i > 0 else 0
        end   = matches[i].start()

        if len(text[start:end].strip()) == 0:  # Sometimes it's confused by punctuation like ?!
            continue

        # Append each word
        sentence = text[start:end]
        x += sentence.split()

        # Append a negative for all the words in the sentence
        y += [OUTPUTS['']] * (len(sentence.split()) - 1)
        y.append(OUTPUTS[matches[i].group()])

    y = [OUTPUTS['.']] + y[:-1]  # Shift the data by one. We want it to guess if punctuation comes *before* the word.

    return x, y
