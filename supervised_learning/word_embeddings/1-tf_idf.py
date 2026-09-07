#!/usr/bin/env python3
"""
Creates a TF-IDF embedding matrix
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding.

    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis.
               If None, all words within sentences should be used.

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: numpy.ndarray of the features used for embeddings
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()

    if hasattr(vectorizer, 'get_feature_names_out'):
        features = vectorizer.get_feature_names_out()
    else:
        features = vectorizer.get_feature_names()

    features = np.array(features)

    return embeddings, features
