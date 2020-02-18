import tensorflow as tf
import math
import numpy as np
from tensorflow.contrib import rnn

class GRU4Rec(object):

    def __init__(self, emb_size, num_usr, num_item, len_Seq, len_Tag, layers = 1, loss_fun = 'BPR', l2_lambda = 0.0):

        self.emb_size = emb_size
        self.item_count = num_item
        self.user_count = num_usr
        self.l2_lambda = l2_lambda
        self.layers = layers
        self.len_Seq = len_Seq
        self.len_Tag = len_Tag
        self.loss_fun = loss_fun

        self.input_Seq = tf.placeholder(tf.int32, [None,self.len_Seq]) #[B,T]
        self.input_NegT = tf.placeholder(tf.int32, [None, None]) #[B,F]
        self.input_PosT = tf.placeholder(tf.int32, [None,None]) #[B]

        self.input_keepprob = tf.placeholder(tf.float32, name='keep_prob')
        self.loss, self.output = self.build_model(self.input_Seq,self.input_NegT
                                                  ,self.input_PosT,self.input_keepprob)

    def build_model(self, in_Seq, in_Neg, in_Pos, in_KP):
        with tf.variable_scope('gru4rec'):
            # Embedding
            self.item_emb = tf.get_variable("item_emb", [self.item_count, self.emb_size]) #[N,e]

            self.W = tf.get_variable("W", [self.item_count, self.emb_size]) #[N,e]
            self.b = tf.get_variable("b", [self.item_count,1])

            session = tf.nn.embedding_lookup(self.item_emb, in_Seq)  # [B,T,e]

            cells = []
            for _ in range(self.layers):
                cell = rnn.GRUCell(self.emb_size, activation=tf.nn.tanh)
                cell = rnn.DropoutWrapper(cell, output_keep_prob=in_KP)
                cells.append(cell)
            self.cell = rnn.MultiRNNCell(cells)

            zero_state = self.cell.zero_state(tf.shape(session)[0], dtype=tf.float32)

            outputs, state = tf.nn.dynamic_rnn(self.cell, session,  initial_state=zero_state)
            output = outputs[:,-1:,:]

            pos_W = tf.nn.embedding_lookup(self.W,in_Pos)
            pos_b = tf.nn.embedding_lookup(self.b,in_Pos)

            neg_W = tf.nn.embedding_lookup(self.W,in_Neg)
            neg_b = tf.nn.embedding_lookup(self.b,in_Neg)

            pos_y = tf.matmul(output, tf.transpose(pos_W,[0,2,1])) + tf.transpose(pos_b,[0,2,1])
            neg_y = tf.matmul(output, tf.transpose(neg_W,[0,2,1])) + tf.transpose(neg_b,[0,2,1])

            if self.loss_fun == 'BPR':
                loss = self.loss_BPR(pos_y,neg_y)
            else:
                loss = self.loss_TOP1(pos_y,neg_y)
            return loss,output



    def loss_BPR(self,pos,neg):
        Ls = -1 * tf.reduce_mean(tf.log(tf.sigmoid(pos - neg)),-1)
        return Ls

    def loss_TOP1(self,pos,neg):
        Ls = tf.reduce_mean(tf.sigmoid(neg - pos) + tf.sigmoid(neg ** 2), -1)
        return Ls

    def predict(self):
        score = tf.matmul(tf.squeeze(self.output), tf.transpose(self.W, [1, 0]))+ tf.transpose(self.b,[1,0])
        score = tf.sigmoid(score)
        return score