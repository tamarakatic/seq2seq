from data_utils import form_ques_answ, clean_text


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

    return form_ques_answ(clean_questions, clean_answers)


def preprocess_twitter_data(lines):
    filter_data = clean_text(lines)
    filter_question, filter_answer = [], []

    for i in range(0, len(filter_data), 2):
        filter_question.append(filter_data[i])
        filter_answer.append(filter_data[i+1])

    return form_ques_answ(filter_question, filter_answer)
