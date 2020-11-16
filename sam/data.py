import pandas as pd

from glob import glob
import os

import spacy

# Global Variables

nlp = spacy.load('en_core_web_sm')
stop_words = nlp.Defaults.stop_words


# Utility/Helper


# API

def data_dir_to_df(dir='../aclImdb/train'):
    """
    Expalin what to do ...
       arg1 -- semantics, what type, etc.
       arg2 -- semantics
    output -- what type, what does it mean
    """
    files_pos = glob(os.path.join(dir, 'pos', '*'))
    files_neg = glob(os.path.join(dir, 'neg', '*'))
    
    data_pos = []
    data_neg = []

    for fname_pos, fname_neg in zip(files_pos, files_neg):
        with open(fname_pos, 'r') as f:
            data_pos.append([f.read(), 1])
        with open(fname_neg, 'r') as f:
            data_neg.append([f.read(), 0])

    data = data_pos + data_neg

    return pd.DataFrame(data, columns=['text', 'label'])


def preprocess(text):
    """
    text -> string
    """
    tokens = [token.text.lower() for token in nlp(text) if token.is_alpha]
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

def generate_vocab(df):
    pass
