import numpy as np
import tensorflow as tf

from data_utils import preprocess_targets
from encoder import encoder_rnn
from decoder import decoder_rnn


def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length,
                  answers_num_words, questions_num_words, encoder_embedding_size,
                  decoder_embedding_size, rnn_size, num_layers, questions_words_to_int):
    encoder_embedding_input = tf.contrib.layers.embed_sequence(inputs,
                                                               answers_num_words + 1,
                                                               encoder_embedding_size,
                                                               initializer=tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedding_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questions_words_to_int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1,
                                                               decoder_embedding_size],
                                                              0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix,
                                                    preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questions_words_to_int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions


def apply_padding(batch_of_sequences, word_to_int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word_to_int['<PAD>']] * (max_sequence_length - len(sequence))
            for sequence in batch_of_sequences]


def split_into_batches(questions, answers, batch_size, questions_words_to_int,
                       answers_words_to_int):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index:start_index + batch_size]
        answers_in_batch = answers[start_index:start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questions_words_to_int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answers_words_to_int))
        yield padded_questions_in_batch, padded_answers_in_batch
