import nltk
import itertools
import numpy as np
import pickle

import tensorflow as tf

from preprocess_data import clean_text

VOCAB_SIZE = 6000


def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, learning_rate, keep_prob


def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([left_side, right_side], axis=1)
    return preprocessed_targets


def convert_string_to_int(question, word2int):
    new_question = ''.join(clean_text(question))
    return [word2int.get(word, word2int['<OUT>']) for word in new_question.split()]


def apply_padding(batch_of_sequences, word_to_int, maxlen):
    pading_idx = []
    for word in batch_of_sequences:
        if word in word_to_int:
            pading_idx.append(word_to_int[word])
        else:
            pading_idx.append(word_to_int['pad'])

    return pading_idx + [0]*(maxlen - len(batch_of_sequences))


def prepare_data(lines):
    filter_data = clean_text(lines)
    filter_questions, filter_answers = [], []

    for i in range(0, len(filter_data), 2):
        question_len = len(filter_data[i].split(' '))
        answer_len = len(filter_data[i+1].split(' '))
        if question_len >= 2 and question_len <= 25:
            if answer_len >= 2 and answer_len <= 25:
                filter_questions.append(filter_data[i])
                filter_answers.append(filter_data[i+1])

    question_tokenized = [wordlist.split(' ') for wordlist in filter_questions]
    answer_tokenized = [wordlist.split(' ') for wordlist in filter_answers]
    tokenized = question_tokenized + answer_tokenized

    frequent_dictionary = nltk.FreqDist(itertools.chain(*tokenized))
    vocabulary = frequent_dictionary.most_common(VOCAB_SIZE)

    ind2word = ['_'] + ['pad'] + [x[0] for x in vocabulary]
    word2idx = dict([(w, i) for i, w in enumerate(ind2word)])

    data_len = len(question_tokenized)
    question_idx = np.zeros([data_len, 25], dtype=np.int32)
    answer_idx = np.zeros([data_len, 25], dtype=np.int32)

    for i in range(data_len):
        question_indexes = apply_padding(question_tokenized[i], word2idx, 25)
        answer_indexes = apply_padding(answer_tokenized[i], word2idx, 25)

        question_idx[i] = np.array(question_indexes)
        answer_idx[i] = np.array(answer_indexes)

    np.save('question_idx.npy', question_idx)
    np.save('answer_idx.npy', answer_idx)

    metadata = {
            'word2idx': word2idx,
            'idx2word': ind2word,
            'freq_dist': frequent_dictionary
            }

    with open('../data/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
