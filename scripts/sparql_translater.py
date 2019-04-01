#!/usr/bin/python
# -*- coding: UTF-8 -*-
import rospy
import numpy as np
from uwds_msgs.msg import SimpleQuery
import keras
from keras.models import Model, Input
from keras.layers import Embedding, LSTM, Bidirectional, Dense
np.random.seed(123)  # for reproducibility


class SparqlTranslater:
    def __init__(self):
        self.data_dir = rospy.get_param("~data_dir")
        self.dataset = rospy.get_param("~dataset")
        self.encoder_input_data = []
        self.decoder_target_data = []
        self.labels_index = {}
        self.labels = []

        self.vocabulary_size = rospy.get_param("~vocabulary_size", 20000)
        self.target_vocabulary_size = 0
        self.embedding_dim = rospy.get_param("~embedding_dim", 128)
        self.latent_dim = rospy.get_param("~latent_dim", 256)
        self.train_src_embedding = rospy.get_param("~train_src_embedding", False)
        self.train_target_embedding = rospy.get_param("~train_target_embedding", False)

        self.encoder_layer = rospy.get_param("~encoder_layer", 2)
        self.decoder_layer = rospy.get_param("~decoder_layer", 2)

        self.nb_inputs = rospy.get_param("~input_size")
        self.nb_outputs = rospy.get_param("~output_size")

        self.batch_size = rospy.get_param("~batch_size", 880)
        self.epochs = rospy.get_param("~epoch", 10000)

        self.model = None

        self.word_to_glove = {}
        self.word_to_index = {}
        self.index_to_word = {}

        self.symbol_to_glove = {}
        self.symbol_to_index = {}
        self.symbol_to_word = {}

        self.nl_lines = []
        self.sparql_queries = []
        self.target_token_index = {}
        self.reverse_target_token_index = {}

        self.loadGloveFile()
        self.loadDataSet()

        self.translate_service = rospy.Service("uwds/nl2sparql", SimpleQuery, self.handleQuery)

    def loadDataSet(self):
        rospy.loginfo("Try to open : '"+self.data_dir+"/geo880-sparql/geo-800-a-sq'")
        tokens_count = {}
        nb_tokens = 0
        with open(self.data_dir+'/geo880-sparql/geo-800-a-sq', 'r') as f:
            for line in f:
                for token in line.split(" "):
                    if token in tokens_count:
                        tokens_count[token] += 1
                    else:
                        self.target_vocabulary_size += 1
            step = 1.0 / nb_tokens+1
            self.target_token_index["\t"] = 1.0
            value = 0 + step
            for symbol, count in tokens_count.items():
                value += step
                self.target_token_index[symbol] = np.array([value], dtype=np.float64)
                self.reverse_target_token_index[value] = symbol

            sequence = []
            for line in f:
                for token in line.split(" "):
                    sequence.append(self.target_token_index[token])
                self.sparql_queries.append(np.array(sequence, dtype=np.float64))

            if len(self.reverse_target_token_index)!=len(self.target_token_index):
                rospy.logerr("Error while creating the sparql symbol tables")

        with open(self.data_dir+'/geo880-sparql/geo-800.en', 'r') as f:
            for line in f:
                self.nl_lines.append(line)
                sequence = []
                for word in line.split(""):
                    sequence.append(self.word_to_glove[word])


    def loadGloveFile(self):
        self.word_to_glove = {}
        self.word_to_index = {}
        self.index_to_word = {}
        rospy.loginfo("Try to open : '"+self.data_dir+"/glove/glove.6B.100d.txt'")
        with open(self.data_dir+'/glove/glove.6B.100d.txt', 'r') as f:
            for line in f:
                record = line.strip().split()
                token = record[0]
                self.word_to_glove[token] = np.array(record[1:], dtype=np.float64)

            tokens = sorted(word_to_glove.keys())
            for idx, tok in enumerate(tokens):
                kerasIdx = idx + 1
                self.word_to_index[tok] = kerasIdx
                self.index_to_word[kerasIdx] = tok
        rospy.loginfo("")

    def loadGloveEmbeddingFromPreTrained(self):
        self.vocabulary_size = len(self.word_to_index) + 1
        self.embedding_dim = next(iter(self.word_to_glove.values())).shape[0]
        self.embedding_matrix = np.zeros((self.vocabulary_size, self.embedding_dim))
        for word, index in self.word_to_index.items():
            self.src_embedding_matrix[index, :] = self.word_to_glove[word]

    def createEncoderDecoder(self):
        # define encoder inputs
        encoder_inputs = Input(shape=(None,))
        # define source embedding
        src_embedding = Embedding(self.vocabulary_size,
                                  self.embedding_dim,
                                  embeddings_initializer=Constant(self.embedding_matrix),
                                  trainable=self.train_src_embedding,
                                  mask_zero=True,
                                  name="source_word_embedding")(encoder_inputs)
        # define encoder
        encoder = Bidirectional(LSTM(self.latent_dim,
                                     return_state=True),
                                name='bidirectional_encoder',
                                merge_mode='concat')

        encoder_outputs, state_h, state_c = encoder(src_embedding)

        encoder_states = [state_h, state_c]
        # define decoder inputs
        decoder_inputs = Input(shape=(None,))
        # define target embedding
        target_embedding = Embedding(self.vocabulary_size,
                                     self.embedding_dim,
                                     embeddings_initializer=Constant(self.embedding_matrix),
                                     trainable=self.train_src_embedding,
                                     mask_zero=True,
                                     name="source_word_embedding")(decoder_inputs)
        # define decoder
        decoder_lstm = Bidirectional(LSTM(self.latent_dim,
                                          return_sequences=True,
                                          return_state=True),
                                     name='bidirectional_decoder',
                                     merge_mode='concat')
        decoder_outputs = decoder_lstm(target_embedding, initial_state=encoder_states)
        decoder_outputs = Dense(self.output_size, activation="softmax")(decoder_outputs)

        # define training model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=0.2)
        self.model.save('uwds.oro.880')

        self.inference_encoder = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        self.inference_decoder = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def translate(self, sentence):
        states_value = self.inference_encoder.predict(sentence)
        target_seq = np.zeros((1, 1, self.output_size))
        target_seq[0, 0, target_token_index['\t']] = 1.
        stop_condition = False
        translated_sentence = ""
        while not stop_condition:
            output_tokens, h, c = self.inference_decoder.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_symbol = reverse_target_char_index[sampled_token_index]
            translated_sentence += sampled_symbol

            if (sampled_symbol == '\n' or len(translated_sentence) > self.output_size):
                stop_condition = True

            target_seq = np.zeros((1, 1, self.output_size))
            target_seq[0, 0, sampled_token_index] = 1.

            states_value = [h, c]

        return translated_sentence

    def handleQuery(req):
        pass


if __name__ == '__main__':
    rospy.init_node("sparql_translater", anonymous=False)
    translater = SparqlTranslater()
    rospy.spin()
