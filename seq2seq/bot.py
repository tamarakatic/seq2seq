import os

import tensorflow as tf
import tensorlayer as tl

from .data_utils import get_metadata
from .constants import MODEL_PATH
from .seq2seq import inferencing_model
from .preprocessing import clean_text
from .utils import color

EMBEDDING_DIM = 1024
SENTENCE_LENGTH = 25
UNK_TOKEN = 'out'


class Bot:

    def __init__(self, model_path=MODEL_PATH):
        (self._start_id,
         self._end_id,
         self._vocab_size,
         self._word2idx,
         self._idx2word) = get_metadata()

        with tl.ops.suppress_stdout():
            (self._encode_seqs,
             self._decode_seqs,
             self._dense_net,
             self._rnn_net,
             self._softmax) = inferencing_model(self._vocab_size, EMBEDDING_DIM)

        self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        tl.layers.initialize_global_variables(self._session)
        tl.files.load_and_assign_npz(sess=self._session,
                                     name=model_path,
                                     network=self._dense_net)

    def respond(self, query):
        query_sequence = self._numericalize_query(query)
        state = self._session.run(self._rnn_net.final_state_encode,
                                  {self._encode_seqs: [query_sequence]})
        sentence = []
        word_id = None
        for _ in range(SENTENCE_LENGTH):
            decode_token = self._start_id if word_id is None else word_id
            out, state = self._session.run([self._softmax, self._rnn_net.final_state_decode],
                                           {self._rnn_net.initial_state_decode: state,
                                            self._decode_seqs: [[decode_token]]})
            word_id = tl.nlp.sample_top(out[0], top_k=2)
            word = self._idx2word[word_id]
            if word_id == self._end_id:
                break
            sentence.append(word)
        return ' '.join([token for token in sentence if token != UNK_TOKEN])

    def _numericalize_query(self, query):
        clean_query = clean_text([query])[0]
        return [self._word2idx.get(word, self._word2idx[UNK_TOKEN])
                for word in clean_query.split()]


if __name__ == '__main__':
    bot = Bot()

    print('\nStarting chat bot. Press CTRL + C to exit.\n')

    try:
        while(True):
            query = input(color("{:>6}".format("Me: "), color='red'))
            response = bot.respond(query)
            print(color('{:>6}'.format('Lucy: '), color='blue') + color(response, color='white'))
            os.system("say -r 0.85 \"{}\"".format(response))

    except (KeyboardInterrupt, EOFError):
        print('\nShutting down')
