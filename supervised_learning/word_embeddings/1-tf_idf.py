#!/usr/bin/env python3
"""
Module defining the tf_idf function
"""
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use, or None

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: list of the features used for embeddings
    """
    cleaned_sentences = []
    for sentence in sentences:
        words = []
        for word in sentence.lower().split():
            cleaned = "".join(c for c in word if c.isalnum())
            if cleaned:
                words.append(cleaned)
        cleaned_sentences.append(words)

    if vocab is None:
        vocab_set = set()
        for words in cleaned_sentences:
            for word in words:
                vocab_set.add(word)
        features = sorted(list(vocab_set))
    else:
        features = sorted(list(set(vocab)))

    s = len(sentences)
    f = len(features)
    tf = np.zeros((s, f), dtype=float)

    for i, words in enumerate(cleaned_sentences):
        if len(words) == 0:
            continue
        for word in words:
            if word in features:
                j = features.index(word)
                tf[i, j] += 1
        tf[i] /= len(words)

    idf = np.zeros(f, dtype=float)
    for j, feature in enumerate(features):
        docs_containing_word = 0
        for words in cleaned_sentences:
            if feature in words:
                docs_containing_word += 1
        if docs_containing_word > 0:
            idf[j] = np.log((s / docs_containing_word)) + 1
        else:
            idf[j] = np.log(s) + 1

    embeddings = tf * idf

    # Normalize or adjust according to standard scikit-learn TF-IDF L2 norm if needed,
    # but here standard raw TF * IDF or standard norm. Let's check standard l2 norm:
    # Holberton's TF-IDF often uses Euclidean (L2) normalization on rows.
    for i in range(s):
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] /= norm

    return embeddings, features
