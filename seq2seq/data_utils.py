import nltk
import itertools
import numpy as np
import pickle

from preprocess_data import clean_text, read_txt, preprocess_cornell_data

OUT = 'out'     # unknown
PAD = 'pad'
MOST_FREQ_WORDS = 10000
MAX_LEN = 25
MIN_LEN = 2


def type_of_data(Twitter=True):
    if Twitter:
        twitter_lines = read_txt('../data/twitter.txt')
        prepare_data(twitter_lines)
    else:
        lines = read_txt('../data/movie_lines.txt')
        conversations = read_txt('../data/movie_conversations.txt')
        preprocess_cornell_data(lines, conversations)


def sequence_list(batch_of_sequences, word_to_int, maxlen):
    pading_idx = []
    for word in batch_of_sequences:
        if word in word_to_int:
            pading_idx.append(word_to_int[word])
        else:
            pading_idx.append(word_to_int[OUT])

    return pading_idx + [0]*(maxlen - len(batch_of_sequences))


def prepare_data(lines):
    filter_data = clean_text(lines)
    filter_questions, filter_answers = [], []

    for i in range(0, len(filter_data), 2):
        question_len = len(filter_data[i].split(' '))
        answer_len = len(filter_data[i+1].split(' '))
        if question_len >= MIN_LEN and question_len <= MAX_LEN:
            if answer_len >= MIN_LEN and answer_len <= MAX_LEN:
                filter_questions.append(filter_data[i])
                filter_answers.append(filter_data[i+1])

    question_tokenized = [wordlist.split(' ') for wordlist in filter_questions]
    answer_tokenized = [wordlist.split(' ') for wordlist in filter_answers]
    tokenized = question_tokenized + answer_tokenized

    frequent_dictionary = nltk.FreqDist(itertools.chain(*tokenized))
    vocabulary = frequent_dictionary.most_common(MOST_FREQ_WORDS)

    ind2word = [PAD] + [OUT] + [x[0] for x in vocabulary]
    word2idx = dict([(w, i) for i, w in enumerate(ind2word)])

    data_len = len(question_tokenized)
    question_idx = np.zeros([data_len, MAX_LEN], dtype=np.int32)
    answer_idx = np.zeros([data_len, MAX_LEN], dtype=np.int32)

    for i in range(data_len):
        question_indexes = sequence_list(question_tokenized[i], word2idx, MAX_LEN)
        answer_indexes = sequence_list(answer_tokenized[i], word2idx, MAX_LEN)

        question_idx[i] = np.array(question_indexes)
        answer_idx[i] = np.array(answer_indexes)

    np.save('../data/question_idx.npy', question_idx)
    np.save('../data/answer_idx.npy', answer_idx)

    metadata = {
            'word2idx': word2idx,
            'idx2word': ind2word,
            'freq_dist': frequent_dictionary
            }

    with open('../data/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)


def load_data(PATH=''):
    try:
        with open(PATH + 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    except:
        metadata = None

    question_idx = np.load(PATH + 'question_idx.npy')
    answer_idx = np.load(PATH + 'answer_idx.npy')
    return metadata, question_idx, answer_idx


def split_dataset(X, y, ratio=[0.7, 0.15, 0.15]):
    data_len = len(X)
    lens = [int(data_len*item) for item in ratio]

    return (X[:lens[0]].tolist(), y[:lens[0]].tolist(),
            X[lens[0]:lens[0]+lens[1]].tolist(), y[lens[0]:lens[0]+lens[1]].tolist(),
            X[-lens[-1]:].tolist(), y[-lens[-1]:].tolist())
