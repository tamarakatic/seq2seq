from preprocess import clean_text


def form_questions_and_answers(lines, conversations):
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
        for i in range(len(conversation) - 1):
            questions.append(id2line[conversation[i]])
            answers.append(id2line[conversation[i + 1]])

    clean_questions = []
    for question in questions:
        clean_questions.append(clean_text(question))

    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_text(answer))

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
