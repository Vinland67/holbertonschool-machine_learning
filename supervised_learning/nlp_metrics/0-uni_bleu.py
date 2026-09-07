#!/usr/bin/env python3
"""
Module to calculate the unigram BLEU score for a sentence
"""
from collections import Counter
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence

    Args:
        references: list of reference translations (lists of words)
        sentence: list containing the model proposed sentence

    Returns:
        the unigram BLEU score
    """
    sent_len = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest_len_idx = np.argmin([abs(r_len - sent_len) for r_len in ref_lens])
    best_ref_len = ref_lens[closest_len_idx]

    sentence_counts = Counter(sentence)
    max_ref_counts = Counter()

    for ref in references:
        ref_counts = Counter(ref)
        for word in sentence_counts:
            max_ref_counts[word] = max(max_ref_counts[word], ref_counts[word])

    clipped_counts = sum(
        min(sentence_counts[word], max_ref_counts[word])
        for word in sentence_counts
    )

    precision = clipped_counts / sent_len if sent_len > 0 else 0

    if sent_len > best_ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - (best_ref_len / sent_len)) if sent_len > 0 else 0

    return bp * precision
