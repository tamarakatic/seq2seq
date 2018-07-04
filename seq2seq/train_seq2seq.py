import time
from sklearn.utils import shuffle

import tensorflow as tf
import tensorlayer as tl

from data_utils import split_dataset, load_data
from seq2seq import training_model, inferencing_model


metadata, question_idx, answer_idx = load_data(PATH='../data/')
epochs = 50
batch_size = 32
embedding_dim = 1024
learning_rate = 0.0001
checkpoint = '../models/checkpoint.npz'
batch_index_check_training_loss = 200
max_sentence_len = 25
list_loss_errors = []
early_stopping_check = 0
early_stopping_stop = 10


def remove_pad_sequences(ques_idx, ans_idx):
    train_X, train_y, test_X, test_y, valid_X, valid_y = split_dataset(ques_idx, ans_idx)
    train_X = tl.prepro.remove_pad_sequences(train_X)
    train_y = tl.prepro.remove_pad_sequences(train_y)
    test_X = tl.prepro.remove_pad_sequences(test_X)
    test_y = tl.prepro.remove_pad_sequences(test_y)
    valid_X = tl.prepro.remove_pad_sequences(valid_X)
    valid_y = tl.prepro.remove_pad_sequences(valid_y)
    return train_X, train_y, test_X, test_y, valid_X, valid_y


def seq2seq_example(idx_to_word_list, train_X, train_y, start_id, end_id):
    print("************************* Seq2seq Example *************************\n")
    print("encode_sequences: ", [idx_to_word_list[id] for id in train_X[10]])
    target_sequences = tl.prepro.sequences_add_end_id([train_y[10]], end_id=end_id)[0]
    print("target_sequences: ", [idx_to_word_list[id] for id in target_sequences])
    decode_sequences = tl.prepro.sequences_add_start_id(
        [train_y[10]], start_id=start_id, remove_last=False)[0]
    print("decode_sequences: ", [idx_to_word_list[id] for id in decode_sequences])
    target_mask = tl.prepro.sequences_get_mask([target_sequences])[0]
    print("target_mask: ", target_mask)
    print("target_sequences len: ", len(target_sequences),
          "decode_sequences len: ", len(decode_sequences),
          "target_mask len: ", len(target_mask))
    print("\n************************* Seq2seq Example *************************")


def get_metadata():
    word_to_idx_dict = metadata['word2idx']
    idx_to_word_list = metadata['idx2word']
    vocab_size = len(metadata['idx2word'])  # 10002
    start_id, end_id = vocab_size, vocab_size + 1  # 10002, 10003
    idx_to_word_list.extend(['start_id', 'end_id'])
    word_to_idx_dict.update({
        'start_id': start_id,
        'end_id': end_id
    })
    return start_id, end_id, vocab_size + 2, word_to_idx_dict, idx_to_word_list


def train_model(train_X, train_y, n_step,
                start_id, end_id,
                X_vocab_size, y_vocab_size,
                word_to_idx_dict, idx_to_word_list):
    encode_seqs_tr, decode_seqs_tr, target_seqs, target_mask, net_out = training_model(
        batch_size,
        X_vocab_size,
        embedding_dim)

    encode_seqs_inf, decode_seqs_inf, net, net_rnn, output = inferencing_model(
        X_vocab_size,
        embedding_dim)

    loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs,
                                               target_seqs=target_seqs,
                                               input_mask=target_mask,
                                               return_details=False,
                                               name='cost')

    net_out.print_params(False)

    train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint, network=net)

    for epoch in range(epochs):
        epoch_time = time.time()

        train_X, train_y = shuffle(train_X, train_y, random_state=0)
        total_err, batch_idx = 0, 0

        for X, y in tl.iterate.minibatches(inputs=train_X,
                                           targets=train_y,
                                           batch_size=batch_size,
                                           shuffle=False):
            step_time = time.time()

            X = tl.prepro.pad_sequences(X)
            _target_seqs = tl.prepro.sequences_add_end_id(y, end_id=end_id)
            _target_seqs = tl.prepro.pad_sequences(_target_seqs)
            _decode_seqs = tl.prepro.sequences_add_start_id(y, start_id=start_id, remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
            _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

            _, error = sess.run([train_optimizer, loss],
                                {encode_seqs_tr: X,
                                 decode_seqs_tr: _decode_seqs,
                                 target_seqs: _target_seqs,
                                 target_mask: _target_mask})

            if batch_idx % batch_index_check_training_loss == 0:
                print('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, Time: {:.2f}s'.format(
                      epoch, epochs,
                      batch_idx, n_step,
                      error,
                      time.time()-step_time))

            total_err += error
            batch_idx += 1

            # inference
            if batch_idx % 1000 == 0:
                ex_queries = [
                    "happy birthday have a nice day",
                    "donald trump won last nights presidential debate according to online polls"
                ]

                for query in ex_queries:
                    print("Lucy > ", query)
                    query_id = [word_to_idx_dict[w] for w in query.split(" ")]
                    for _ in range(5):      # 5 answers for 1 query
                        # encode => get state
                        state = sess.run(net_rnn.final_state_encode,
                                         {encode_seqs_inf: [query_id]})
                        # decode => feed start_id, get first word
                        out, state = sess.run([output, net_rnn.final_state_decode],
                                              {net_rnn.initial_state_decode: state,
                                               decode_seqs_inf: [[start_id]]})
                        word_id = tl.nlp.sample_top(out[0], top_k=3)
                        word = idx_to_word_list[word_id]
                        # decode => feed state iteratively
                        sentence = [word]
                        for _ in range(max_sentence_len):
                            out, state = sess.run([output, net_rnn.final_state_decode],
                                                  {net_rnn.initial_state_decode: state,
                                                   decode_seqs_inf: [[word_id]]})
                            word_id = tl.nlp.sample_top(out[0], top_k=3)
                            word = idx_to_word_list[word_id]
                            if word_id == end_id:
                                break
                            sentence = sentence + [word]
                        print(" >", ' '.join(sentence))
        average_loss_error = total_err/batch_idx
        print('Epoch: {}/{}, Averaged Loss: {:.4f}, Time: {:.2f}s'.format(
              epoch, epochs,
              average_loss_error,
              time.time() - epoch_time))

        list_loss_errors.append(average_loss_error)
        if average_loss_error <= min(list_loss_errors):
            print("I speak better now!!!")
            early_stopping_check = 0
            tl.files.save_npz(net.all_params, name=checkpoint, sess=sess)
        else:
            print("Sorry I don't speak better. I need to practice more!!!")
            early_stopping_check += 1
            if early_stopping_check == early_stopping_stop:
                break


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, valid_X, valid_y = remove_pad_sequences(
        question_idx, answer_idx)
    X_seq_len, y_seq_len = len(train_X), len(train_y)
    assert X_seq_len == y_seq_len

    n_step = int(X_seq_len/batch_size)      # 7708
    start_id, end_id, X_vocab_size, y_vocab_size, \
        word_to_idx_dict, idx_to_word_list = prepare_parameters(train_X, train_y)
    train_model(train_X, train_y, n_step,
                start_id, end_id,
                X_vocab_size, y_vocab_size,
                word_to_idx_dict, idx_to_word_list)
