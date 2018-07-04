import codecs
import re

EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)


def read_txt(path):
    lines = codecs.open(path, 'r', encoding='utf-8', errors='ignore').read().split('\n')
    if len(lines) % 2 != 0:
        lines = lines[:-1]
    return lines


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
        line = " ".join(line.split())
        clean_lines.append(line)
    return clean_lines


def preprocess_cornell_data(lines, conversations):
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]

    conversations_ids = []
    for conversation in conversations[:-1]:
        _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        conversations_ids.append(_conversation.split(','))

    questions = []
    answers = []
    for conversation in conversations_ids:
        for i in range(0, len(conversation), 2):
            questions.append(id2line[conversation[i]])
            answers.append(id2line[conversation[i + 1]])

    clean_questions = []
    for question in questions:
        clean_questions.append(clean_text(question))

    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_text(answer))

    return clean_questions, clean_answers
