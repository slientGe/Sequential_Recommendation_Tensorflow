from modules import *
import pandas as pd
import tensorflow as tf


class SASRec():
    def __init__(self, usernum = 100,
                 itemnum = 100,
                 emb_size = 128,
                 max_Seqlens = 50,
                 num_blocks = 1,
                 num_heads = 1,
                 dropout_rate = 0.8,
                 reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.max_len))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.max_len))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.max_len))
        self.usernum = usernum
        self.itemnum = itemnum
        self.reuse = reuse
        self.hidden_units = emb_size
        self.l2_emb = 1e-6
        self.maxlen = max_Seqlens
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        # pos = self.pos
        # neg = self.neg


        self.loss,self.seq_emb = self.build_network(self.u,self.input_seq,self.pos,self.neg)



    def build_network(self,u,input_seq,pos,neg,is_training):
        mask = tf.expand_dims(tf.to_float(tf.not_equal(input_seq, 0)), -1)

        self.input_seq = input_seq

        with tf.variable_scope("SASRec", reuse=self.reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(input_seq,
                                                 vocab_size=self.itemnum,
                                                 num_units=self.hidden_units,
                                                 zero_pad=False,
                                                 scale=True,
                                                 l2_reg=self.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=None
                                                 )
            self.item_emb_table = item_emb_table

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0), [tf.shape(input_seq)[0], 1]),
                vocab_size=self.maxlen,
                num_units=self.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=self.l2_emb,
                scope="dec_pos",
                reuse=self.reuse,
                with_t=True
            )
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=self.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=self.dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)

        pos = tf.reshape(pos, [tf.shape(input_seq)[0] * self.maxlen])
        neg = tf.reshape(neg, [tf.shape(input_seq)[0] * self.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(input_seq)[0] * self.maxlen, self.hidden_units])

        self.seq_emb  = seq_emb

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(input_seq)[0] * self.maxlen])


        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        return self.loss, self.seq_emb

        # tf.summary.scalar('loss', self.loss)
        # self.auc = tf.reduce_sum(
        #     ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        # ) / tf.reduce_sum(istarget)
        #
        # if reuse is None:
        #     tf.summary.scalar('auc', self.auc)
        #     self.global_step = tf.Variable(0, name='global_step', trainable=False)
        #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta2=0.98)
        #     self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        # else:
        #     tf.summary.scalar('test_auc', self.auc)
        #
        # self.merged = tf.summary.merge_all()






    def predict(self,item_list):
        len_item = len(item_list)
        all_index = tf.convert_to_tensor(item_list, dtype=tf.int32)
        test_item_emb = tf.nn.embedding_lookup(self.item_emb_table, all_index)
        self.test_logits = tf.matmul(self.seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], self.maxlen, len_item])
        self.test_logits = self.test_logits[:, -1, :]

        return self.test_logits

        #
        #
        # return sess.run(self.test_logits,
        #                 {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})
