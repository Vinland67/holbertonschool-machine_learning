#!/usr/bin/env python3
"""
Module defining the word2vec_model function using gensim
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim word2vec model

    Args:
        sentences: list of sentences to be trained on
        vector_size: dimensionality of the embedding layer
        min_count: minimum number of occurrences of a word
        window: maximum distance between current and predicted word
        negative: size of negative sampling
        cbow: boolean; True for CBOW, False for Skip-gram
        epochs: number of iterations to train over
        seed: seed for the random number generator
        workers: number of worker threads

    Returns:
        trained gensim Word2Vec model
    """
    sg = 0 if cbow else 1

    # Modeli sentences parametri olmadan yaradırıq ki, avtomatik təlim olmasın
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        seed=seed,
        epochs=epochs
    )

    # Lüğəti qururuq
    model.build_vocab(sentences)

    # Modeli yalnız bir dəfə nəzarətli şəkildə təlim edirik
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )

    return model
