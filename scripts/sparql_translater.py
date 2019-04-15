#!/usr/bin/python
# -*- coding: UTF-8 -*-
import rospy
import numpy as np
from uwds_msgs.srv import SimpleQuery
# import keras
# from keras.models import Model, Input
# from keras.layers import Embedding, LSTM, Bidirectional, Dense
np.random.seed(123)  # for reproducibility


class SparqlTranslater:
    def __init__(self):
        self.data_dir = rospy.get_param("~data_dir", "")
        self.glove_path = rospy.get_param("~glove_path", self.data_dir+"/glove/glove.6B.100d.txt")
        self.input_dataset_path = rospy.get_param("~input_dataset", self.data_dir+"/uwds_dataset/uwds.en")
        self.output_dataset_path = rospy.get_param("~output_dataset", self.data_dir+"/uwds_dataset/uwds.sq")
        self.symbols_path = rospy.get_param("~symbols_path", self.data_dir+"/uwds_dataset/symbols.txt")

        self.model = None

        self.word_to_vector = {}
        self.word_to_index = {}
        self.index_to_word = {}

        weights_path = rospy.get_param("~weights_path", self.data_dir)

        try:
            self.model = self.create_model(self.glove_path)
            model.load_weights(weights_path)
        except:
            self.model = self.create_model(self.glove_path)
            self.X_train, self.y_train = self.load_training_dataset(self.input_dataset_path, self.output_dataset_path)
            #self.train()
            # checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
            # model.fit(self.X_train, self.y_train, nb_epoch=20, batch_size=16, show_accuracy=True, validation_split=0.2, verbose=2, callbacks=[checkpointer])

        # rospy.loginfo(self.decode(self.encode("hello guys how are you ?")))
        # rospy.loginfo(self.encode("hello guys how are you ?"))
        # self.translate_service = rospy.Service("uwds/nl2sparql", SimpleQuery, self.handleQuery)

    def create_model(self, embedding_path):
        # load embedding
        rospy.loginfo("Start loading glove embedding...")
        self.word_to_vector, self.word_to_index, self.index_to_word = self.load_glove_file(embedding_path)
        # load additionnal symbols
        original_dict_size = len(self.word_to_index)
        rospy.loginfo("Dictionnary size : "+str(original_dict_size))
        original_embedding_size = next(iter(self.word_to_vector.values())).shape[0]
        rospy.loginfo("Original embedding dim : "+str(original_embedding_size))

        new_symbols = self.load_additional_symbols(self.symbols_path, original_embedding_size)

        # for symbol in new_symbols:
        rospy.loginfo("Nb of new symbols to add : "+str(len(new_symbols)))

        new_embedding_size = original_embedding_size + len(new_symbols)

        rospy.loginfo("Adding them as hot encoding...")
        for word, vector in self.word_to_vector.items():
            # add new dim for
            self.word_to_vector[word] = np.append(vector, np.zeros(len(new_symbols)))
        index = 0
        for symbol in new_symbols:
            self.word_to_vector[symbol] = np.zeros(new_embedding_size)
            self.word_to_vector[symbol][original_embedding_size+index] = 1.0
            self.word_to_index[symbol] = original_dict_size+index
            self.index_to_word[original_dict_size+index] = symbol
            index+=1

        new_embedding_size = original_embedding_size + len(new_symbols)
        rospy.loginfo("New embedding dim : "+str(new_embedding_size))
        rospy.loginfo("New dictionnary size : "+str(original_dict_size+len(new_symbols)))

    def load_additional_symbols(self, symbols_path, source_embedding_dim):
        new_symbols = []
        with open(symbols_path, 'r') as f:
            for line in f:
                new_symbols.append(line[:len(line)-1])
        return new_symbols

    def load_glove_file(self, embedding_path):
        word_to_glove = {}
        word_to_index = {}
        index_to_word = {}
        rospy.loginfo("Try to open : '"+embedding_path)
        with open(embedding_path, 'r') as f:
            for line in f:
                record = line.strip().split()
                token = record[0]
                word_to_glove[token] = np.array(record[1:], dtype=np.float64)

            tokens = sorted(word_to_glove.keys())
            for idx, tok in enumerate(tokens):
                kerasIdx = idx + 1
                word_to_index[tok] = kerasIdx
                index_to_word[kerasIdx] = tok

        return word_to_glove, word_to_index, index_to_word

    def load_training_dataset(self, input_path, output_path):
        input_sentences = []
        output_sentences = []
        input_train = []
        output_train = []

        with open(input_path, 'r') as f:
            for line in f:
                input_sentences.append(line[:len(line)-1])

        with open(output_path, 'r') as f:
            for line in f:
                output_sentences.append(line[:len(line)-1])

        if len(input_sentences) != len(output_sentences):
            rospy.logerr("Dataset incorrect !")
            return None

        for i in range(0, len(input_sentences)):
            input_train.append(self.encode_vector(input_sentences[i]))
            output_train.append(self.encode_vector(output_sentences[i]))
            rospy.loginfo(str(input_sentences[i])+" => "+str(self.encode_seq(input_sentences[i])))
            rospy.loginfo(str(output_sentences[i])+" => "+str(self.encode_seq(output_sentences[i])))

        print input_train
        X_train = np.array(input_train, dtype=np.float64)
        y_train = np.array(output_train, dtype=np.float64)
        X_train.reshape((1, X_train.shape[0], X_train.shape[1]))
        y_train.reshape((1, y_train.shape[0], y_train.shape[1]))
        return X_train, y_train

    def encode_seq(self, sentence):
        sequence = []
        for symbol in sentence.split(" "):
            if symbol in self.word_to_index:
                sequence.append(self.word_to_index[symbol])
            else:
                rospy.logerr("Word '"+str(symbol)+"' not present in the dictionnary")
        return sequence

    def encode_vector(self, sentence):
        sequence = []
        for symbol in sentence.split(" "):
            if symbol in self.word_to_vector:
                sequence.append(self.word_to_vector[symbol])
            else:
                rospy.logerr("Word '"+str(symbol)+"' not present in the dictionnary")
        return sequence

    def decode(self, sequence):
        sentence = ""
        first_word = True
        for index in sequence:
            if index in self.index_to_word:
                if first_word is True:
                    sentence += str(self.index_to_word[index])
                    first_word = False
                else:
                    sentence += " " + str(self.index_to_word[index])
            else:
                rospy.logerr("Index '"+str(index)+"' not present in the dictionnary")
        return sentence
    #
    # def loadGloveEmbeddingFromPreTrained(self):
    #     self.vocabulary_size = len(self.word_to_index) + 1
    #     self.embedding_dim = next(iter(self.word_to_glove.values())).shape[0]
    #     self.embedding_matrix = np.zeros((self.vocabulary_size, self.embedding_dim))
    #     for word, index in self.word_to_index.items():
    #         self.src_embedding_matrix[index, :] = self.word_to_glove[word]
    #
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
        decoder = Bidirectional(LSTM(self.latent_dim,
                                          return_sequences=True,
                                          return_state=True),
                                     name='bidirectional_decoder',
                                     merge_mode='concat')
        decoder_outputs = decoder(target_embedding, initial_state=encoder_states)
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

    # def translate(self, sentence):
    #     states_value = self.inference_encoder.predict(sentence)
    #     target_seq = np.zeros((1, 1, self.output_size))
    #     target_seq[0, 0, target_token_index['\t']] = 1.
    #     stop_condition = False
    #     translated_sentence = ""
    #     while not stop_condition:
    #         output_tokens, h, c = self.inference_decoder.predict([target_seq] + states_value)
    #         sampled_token_index = np.argmax(output_tokens[0, -1, :])
    #         sampled_symbol = reverse_target_char_index[sampled_token_index]
    #         translated_sentence += sampled_symbol
    #
    #         if (sampled_symbol == '\n' or len(translated_sentence) > self.output_size):
    #             stop_condition = True
    #
    #         target_seq = np.zeros((1, 1, self.output_size))
    #         target_seq[0, 0, sampled_token_index] = 1.
    #
    #         states_value = [h, c]
    #
    #     return translated_sentence
    #
    # def handleQuery(req):
    #     pass


if __name__ == '__main__':
    rospy.init_node("sparql_translater", anonymous=False)
    translater = SparqlTranslater()
    rospy.spin()
