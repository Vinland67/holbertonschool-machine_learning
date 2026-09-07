#!/usr/bin/env python3
"""
Module defining the bag_of_words function
"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use, or None

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: list of the features used for embeddings
    """
    cleaned_sentences = []
    for sentence in sentences:
        # Lowercase and split by non-alphanumeric, filtering out empty strings
        words = []
        current_word = []
        for char in sentence.lower():
            if char.isalnum():
                current_word.append(char)
            else:
                if current_word:
                    words.append("".join(current_word))
                    current_word = []
        if current_word:
            words.append("".join(current_word))
        cleaned_sentences.append(words)

    if vocab is None:
        vocab_set = set()
        for words in cleaned_sentences:
            for word in words:
                vocab_set.add(word)
        features = sorted(list(vocab_set))
    else:
        features = sorted(list(set(vocab)))

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in features:
                j = features.index(word)
                embeddings[i, j] += 1

    return embeddings, features
