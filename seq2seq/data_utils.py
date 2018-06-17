import codecs
import re

import tensorflow as tf

EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)


def read_txt(path):
    return codecs.open(path, 'r', encoding='utf-8', errors='ignore').read().split('\n')[:-1]


def clean_text(lines):
    clean_lines = []
    for line in lines:
        line = line.lower()
        line = re.sub(r"i'm", "i am", line)
        line = re.sub(r"he's", "he is", line)
        line = re.sub(r"she's", "she is", line)
        line = re.sub(r"that's", "that is", line)
        line = re.sub(r"what's", "what is", line)
        line = re.sub(r"where's", "where is", line)
        line = re.sub(r"how's", "how is", line)
        line = re.sub(r"\'ll", " will", line)
        line = re.sub(r"\'ve", " have", line)
        line = re.sub(r"\'re", " are", line)
        line = re.sub(r"\'d", " would", line)
        line = re.sub(r"n't", " not", line)
        line = re.sub(r"won't", "will not", line)
        line = re.sub(r"can't", "cannot", line)
        line = EMOJI.sub(r'', line)
        line = re.sub(r"[-()\"#/%&$@;:<>*^_{}`+=~|.!?,]", "", line)
        clean_lines.append(line)
    return clean_lines


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
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]


def form_ques_answ(clean_questions, clean_answers):
    short_questions = []
    short_answers = []
    i = 0
    for question in clean_questions:
        if 2 <= len(question.split()) <= 25:
            short_questions.append(question)
            short_answers.append(clean_answers[i])
        i += 1

    clean_questions = []
    clean_answers = []
    i = 0
    for answer in short_answers:
        if 2 <= len(answer.split()) <= 25:
            clean_answers.append(answer)
            clean_questions.append(short_questions[i])
        i += 1

    word2count = {}
    for question in clean_questions:
        for word in question.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

    for answer in clean_answers:
        for word in answer.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

    threshold = 15
    questions_words_to_int = {}
    word_number = 0
    for word, count in word2count.items():
        if count >= threshold:
            questions_words_to_int[word] = word_number
            word_number += 1

    answers_words_to_int = {}
    word_number = 0
    for word, count in word2count.items():
        if count >= threshold:
            answers_words_to_int[word] = word_number
            word_number += 1

    tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
    for token in tokens:
        questions_words_to_int[token] = len(questions_words_to_int) + 1

    for token in tokens:
        answers_words_to_int[token] = len(answers_words_to_int) + 1

    answers_ints_to_word = {w_i: w for w, w_i in answers_words_to_int.items()}

    for i in range(len(clean_answers)):
        clean_answers[i] += ' <EOS>'

    questions_into_int = []
    for question in clean_questions:
        ints = []
        for word in question.split():
            if word not in questions_words_to_int:
                ints.append(questions_words_to_int['<OUT>'])
            else:
                ints.append(questions_words_to_int[word])
        questions_into_int.append(ints)

    answers_into_int = []
    for answer in clean_answers:
        ints = []
        for word in answer.split():
            if word not in answers_words_to_int:
                ints.append(answers_words_to_int['<OUT>'])
            else:
                ints.append(answers_words_to_int[word])
        answers_into_int.append(ints)

    sorted_clean_questions = []
    sorted_clean_answers = []
    for length in range(1, 25 + 1):
        for i in enumerate(questions_into_int):
            if len(i[1]) == length:
                sorted_clean_questions.append(questions_into_int[i[0]])
                sorted_clean_answers.append(answers_into_int[i[0]])

    return (answers_words_to_int,
            answers_ints_to_word,
            questions_words_to_int,
            sorted_clean_questions,
            sorted_clean_answers)
