# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:14:57 2021

@author: Fabretto

Defines functions for the Project 6 'Catégorisez automatiquement des questions'

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

from collections import defaultdict

import regex
import nltk, pprint
from nltk import word_tokenize, regexp_tokenize


def to_text(token_chain):
    '''
    Change tokens chained with '|' to a text of tokens separated by a space
    '''
    return ' '.join(token for token in token_chain.split(sep='|'))

def to_list(token_chain):
    '''
    Change tokens chained with '|' to a list of tokens
    '''
    return token_chain.split(sep='|')

def freq_stats_corpora(ids_text, tokenizer, top_nstop=None):
    """ Tokenize and normalize a set of documents ids_text.
        Discards most common words from document set and nltk list
        of English stop words. Discards also words less than 3
        characters long.
        
        input:
            ids_text: dataframe(ids, text)
                ids: doc identifier
                text: text
            tokenizer:  a tokenizer instance
            top_nstop: number of most common words to add to the stop words
        output:
            freq: frequency distribution by doc
            stats: total and unique word count by doc
            corpora: tokenized and normalized corpus
            sw: stop words discarded from corpora
            freq_totale: total frequency distribution
    """
    corpora = defaultdict(list)
    porter = nltk.PorterStemmer()

    # Create a corpus of tokens by query
    for qid, text in ids_text.itertuples(index=False):
        corpora[qid] = tokenizer.tokenize(text.lower())

    stats, freq = dict(), dict()
    freq_totale = nltk.Counter()

    # Compute total frequency, including stop words
    for v in corpora.values():
        freq_totale += nltk.FreqDist(v)

    most_common = [] if top_nstop is None else list(
        zip(*freq_totale.most_common(top_nstop)))[0]

    sw = set()
    sw.update(most_common)
    sw.update(tuple(nltk.corpus.stopwords.words('english')))
    
    freq_totale.clear()

    for k, v in corpora.items():
        corpora[k] = tokens = [
            porter.stem(w) for w in v if not w in list(sw) and len(w) >= 3
        ]
        freq[k] = fq = nltk.FreqDist(tokens)
        freq_totale += fq
        stats[k] = {'total': fq.N(), 'unique': fq.B()}

    return (freq, stats, corpora, sw, freq_totale)

def freq_stats_wtags(ids_text, tokenizer):
    """
    Tokenize and normalize a set of Tags ids_text into words for grouping.
        
        input:
            ids_text: dataframe(ids, text)
                ids: doc identifier
                text: tags
                tokenizer: a tokenizer instance
        output:
            freq: frequency distribution by doc
            stats: total and unique word count by doc
            corpora: tokenized and normalized tags
            freq_totale:
     """
    corpora = defaultdict(list)

    # Create a corpus of tokens by query
    for qid, text in ids_text.itertuples(index=False):
        corpora[qid] = tokenizer.tokenize(text.lower())

    stats, freq = dict(), dict()
    freq_totale = nltk.Counter()

    # Compute total frequency, including stop words
    for v in corpora.values():
        freq_totale += nltk.FreqDist(v)

    for k, v in corpora.items():
        freq[k] = fq = nltk.FreqDist(v)
        freq_totale += fq
        stats[k] = {'total': fq.N(), 'unique': fq.B()}

    return (freq, stats, corpora, freq_totale)

def tokenize_raw(raw_text, tokenizer, vocabulary):
    """ Tokenize and normalize a raw text. Output only tokens
        out of a given vocabulary.
        Discards words from nltk list of English stop words. Discards also
        words less than 3 characters long.
        
        input:
            raw_text: raw text, typically a query title
            tokenizer:  a tokenizer instance
            vocabulary: output only tokens from this vocabulary
        output:
            token_string: tokenized and normalized string from given vocabulary
    """
    porter = nltk.PorterStemmer()

    sw = set()
    sw.update(tuple(nltk.corpus.stopwords.words('english')))
    
    tokens = [
        porter.stem(w) for w in tokenizer.tokenize(raw_text.lower())
        if not w in list(sw) and len(w) >= 3
    ]

    return ' '.join([token for token in tokens if token in vocabulary])

