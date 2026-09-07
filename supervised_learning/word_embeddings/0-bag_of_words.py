#!/usr/bin/env python3
"""
Creates a bag of words embedding matrix
"""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis.
               If None, all words within sentences should be used.

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: numpy.ndarray of the features used for embeddings
    """
    sents_words = []
    for sentence in sentences:
        s = sentence.lower()
        s = re.sub(r"'s\b", "", s)
        words = re.findall(r'\b\w+\b', s)
        sents_words.append(words)

    if vocab is None:
        unique_words = set()
        for words in sents_words:
            unique_words.update(words)
        vocab = sorted(list(unique_words))

    features = np.array(vocab)
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    vocab_dict = {w: i for i, w in enumerate(vocab)}

    for i, words in enumerate(sents_words):
        for word in words:
            if word in vocab_dict:
                embeddings[i, vocab_dict[word]] += 1

    return embeddings, features
