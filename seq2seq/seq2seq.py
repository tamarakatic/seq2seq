import matplotlib as mpl
mpl.use('TkAgg')

import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers import EmbeddingInputlayer
from tensorlayer.layers import retrieve_seq_length_op2
from tensorlayer.layers import Seq2Seq
from tensorlayer.layers import DenseLayer


def seq2seq_model(encode_sequences,
                  decode_sequences,
                  vocabulary_size,
                  embedding_dim,
                  is_train=True,
                  reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                        inputs=encode_sequences,
                        vocabulary_size=vocabulary_size,
                        embedding_size=embedding_dim,
                        name='seq_embedding')
            vs.reuse_variables()
            tl.layers.set_name_reuse(True)
            net_decode = EmbeddingInputlayer(
                inputs=decode_sequences,
                vocabulary_size=vocabulary_size,
                embedding_size=embedding_dim,
                name='seq_embedding')
        net_rnn = Seq2Seq(net_encode, net_decode,
                          cell_fn=tf.contrib.rnn.BasicLSTMCell,
                          n_hidden=embedding_dim,
                          initializer=tf.random_uniform_initializer(-0.1, 0.1),
                          encode_sequence_length=retrieve_seq_length_op2(encode_sequences),
                          decode_sequence_length=retrieve_seq_length_op2(decode_sequences),
                          initial_state_encode=None,
                          dropout=(0.5 if is_train else None),
                          n_layer=3,
                          return_seq_2d=True,
                          name='seq2seq')
        net_out = DenseLayer(net_rnn, n_units=vocabulary_size, act=tf.identity, name='output')
    return net_out, net_rnn


def training_model(batch_size, vocab_size, embed_dim):
    encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
    decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
    target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
    target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask")

    net_out, _ = seq2seq_model(encode_seqs, decode_seqs, vocab_size, embed_dim, is_train=True, reuse=False)
    return encode_seqs, decode_seqs, target_seqs, target_mask, net_out


def inferencing_model(vocab_size, embed_dim):
    encode_seqs = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
    decode_seqs = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")

    net, net_rnn = seq2seq_model(encode_seqs, decode_seqs, vocab_size, embed_dim, is_train=False, reuse=True)
    output = tf.nn.softmax(net.outputs)
    return encode_seqs, decode_seqs, net, net_rnn, output
