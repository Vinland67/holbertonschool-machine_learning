#!/usr/bin/env python3
"""
Module defining the question_answer function using BERT for QA
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question

    Args:
        question: string containing the question to answer
        reference: string containing the reference document

    Returns:
        string containing the answer, or None if no answer is found
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + reference_tokens + ['[SEP]']

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (len(question_tokens) + 2) + [1] * (len(reference_tokens) + 1)

    input_word_ids = tf.expand_dims(tf.convert_to_tensor(input_word_ids, dtype=tf.int32), 0)
    input_mask = tf.expand_dims(tf.convert_to_tensor(input_mask, dtype=tf.int32), 0)
    input_type_ids = tf.expand_dims(tf.convert_to_tensor(input_type_ids, dtype=tf.int32), 0)

    outputs = model([input_word_ids, input_mask, input_type_ids])

    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    if short_start > short_end:
        return None

    answer_tokens = tokens[short_start:short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if not answer.strip():
        return None

    return answer
