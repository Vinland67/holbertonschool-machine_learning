#!/usr/bin/env python3
"""
Module to calculate the cumulative n-gram BLEU score for a sentence
"""
from collections import Counter
import numpy as np


def _get_ngrams(tokens, n):
    """Generates n-grams from a list of tokens."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence

    Args:
        references: list of reference translations (lists of words)
        sentence: list containing the model proposed sentence
        n: size of the largest n-gram to use for evaluation

    Returns:
        the cumulative n-gram BLEU score
    """
    sent_len = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest_len_idx = np.argmin([abs(r_len - sent_len) for r_len in ref_lens])
    best_ref_len = ref_lens[closest_len_idx]

    precisions = []

    for i in range(1, n + 1):
        sent_ngrams = _get_ngrams(sentence, i)
        sent_counts = Counter(sent_ngrams)

        max_ref_counts = Counter()
        for ref in references:
            ref_ngrams = _get_ngrams(ref, i)
            ref_counts = Counter(ref_ngrams)
            for ngram in sent_counts:
                max_ref_counts[ngram] = max(
                    max_ref_counts[ngram], ref_counts[ngram]
                )

        clipped_counts = sum(
            min(sent_counts[ngram], max_ref_counts[ngram])
            for ngram in sent_counts
        )

        total_sent_ngrams = len(sent_ngrams)
        if total_sent_ngrams == 0:
            precision = 0
        else:
            precision = clipped_counts / total_sent_ngrams

        precisions.append(precision)

    if sent_len > best_ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - (best_ref_len / sent_len)) if sent_len > 0 else 0

    if any(p == 0 for p in precisions):
        geometric_mean = 0.0
    else:
        weight = 1.0 / n
        geometric_mean = np.exp(
            sum(weight * np.log(p) for p in precisions)
        )

    return bp * geometric_mean
