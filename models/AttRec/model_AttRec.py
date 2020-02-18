import tensorflow as tf
import math
import numpy as np

class AttRec(object):
    def __init__(self, emb_size, num_usr, num_item, len_Seq, len_Tag, score_weight = 0.3, gamma = 0.5, l2_lambda = 0.0):

        self.emb_size = emb_size

        self.item_count = num_item
        self.user_count = num_usr
        self.w = score_weight
        self.gamma = gamma
        self.l2_lambda = l2_lambda
        self.len_Seq = len_Seq
        self.len_Tag = len_Tag,


        self.u_init = tf.keras.initializers.he_normal()
        self.init = tf.random_normal_initializer(0, 0.5 / self.emb_size)

        self.input_Seq = tf.placeholder(tf.int32, [None, None]) #[B,T]
        self.input_Usr = tf.placeholder(tf.int32, [None]) #[B]
        self.input_NegT = tf.placeholder(tf.int32, [None, None]) #[B,F]
        self.input_PosT = tf.placeholder(tf.int32, [None,None]) #[B]

        self.input_keepprob = tf.placeholder(tf.float32, name='keep_prob')



        self.loss, self.m = self.build_model(self.input_Seq,self.input_Usr
                                             ,self.input_NegT,self.input_PosT,self.input_keepprob)

    def build_model(self, in_Seq, in_Usr, in_Neg, in_Pos, in_KP):
        with tf.variable_scope('AttRec'):
            # Embedding
            self.item_emb = item_emb = \
                tf.get_variable("item_emb", [self.item_count, self.emb_size],initializer=self.u_init) #[N,e]
            self.item_rep_emb = item_rep_emb = \
                tf.get_variable("item_rep_emb", [self.item_count, self.emb_size],initializer=self.init) #[N,e]
            self.user_rep_emb = user_reo_emb = \
                tf.get_variable("user_rep_emb", [self.user_count, self.emb_size],initializer=self.init) #[N,e]


            value = tf.nn.embedding_lookup(self.item_emb, in_Seq, max_norm=1)  # [B,T,e]

            # Self Attention

            ts = self._make_time_signal(self.emb_size, self.len_Seq) #[msl, e]
            ts = tf.expand_dims(ts, 0)

            key = query = tf.add(value, ts) #[B,T,e]
            att_output = self._attention_module(query, key, value,self.emb_size,in_KP)

            # Mean
            m = tf.reduce_mean(att_output,axis=1)

            #Look up embedding for targets and negative samples

            u = tf.nn.embedding_lookup(self.user_rep_emb, in_Usr) #[B,e]
            u = tf.clip_by_norm(u, 1, -1)

            pos_v = tf.nn.embedding_lookup(self.item_rep_emb, in_Pos) #[B,pos,e]
            pos_v = tf.clip_by_norm(pos_v,1,-1)

            neg_v = tf.nn.embedding_lookup(self.item_rep_emb, in_Neg) #[B,neg,e]
            neg_v = tf.clip_by_norm(neg_v, 1, -1)

            pos_x = tf.nn.embedding_lookup(self.item_emb, in_Pos) #[B,pos,e]
            pos_x = tf.clip_by_norm(pos_x, 1, -1)

            neg_x = tf.nn.embedding_lookup(self.item_emb, in_Neg)#[B,neg,e]
            neg_x = tf.clip_by_norm(neg_x, 1, -1)

            pos_y = self._pos_object_function(u, pos_v, m, pos_x, self.w) #[B,tsl]
            neg_y = self._neg_object_function(u, neg_v, m, neg_x, self.w) #[B,nsl]

            #margin based hinge loss
            loss = self.loss_function(self.gamma, self.l2_lambda, pos_y, neg_y)
            next_items = m

            return loss,next_items

    def predict(self, item_list):

        all_idx = tf.convert_to_tensor(item_list,dtype=tf.int32)

        u = tf.nn.embedding_lookup(self.user_rep_emb, self.input_Usr)
        U = tf.tile(tf.expand_dims(u, [1]), [1, self.item_count, 1])
        m = tf.tile(tf.expand_dims(self.m, [1]), [1, self.item_count, 1])
        item_r = tf.nn.embedding_lookup(self.item_rep_emb, all_idx, max_norm=1)
        item_e = tf.nn.embedding_lookup(self.item_emb, all_idx, max_norm=1)

        score = self.w * tf.reduce_sum(tf.square(U - item_r), axis=-1) + (1 - self.w) * tf.reduce_sum(
            tf.square(m - item_e), axis=-1)

        return score


    def _attention_module(self,query,key,value,unit,in_kp):
        with tf.variable_scope('attention',reuse=True):
            query = tf.layers.dense(query, unit, name='qk_map', activation=tf.nn.relu, use_bias=False,kernel_initializer=self.u_init,
                                    reuse=tf.AUTO_REUSE,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
            query = tf.nn.dropout(query,in_kp)

            key = tf.layers.dense(key, self.emb_size, name='qk_map', activation=tf.nn.relu, use_bias=False,kernel_initializer=self.u_init,
                                  reuse=tf.AUTO_REUSE,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
            key = tf.nn.dropout(key, in_kp)

            score = tf.matmul(query, tf.transpose(key, [0, 2, 1])) / math.sqrt(self.emb_size)  # [B,T,T]

            #masks the diagonal of the affinity matrix
            a_mask = tf.ones([tf.shape(score)[1], tf.shape(score)[2]])
            a_mask = a_mask - tf.matrix_diag(tf.ones([tf.shape(score)[1]]))
            a_mask = tf.expand_dims(a_mask, [0])
            a_mask = tf.tile(a_mask, [tf.shape(score)[0], 1, 1])
            score *= a_mask
            score = tf.nn.softmax(score, axis=2)
            output = tf.matmul(score, value)
            return output



    def _pos_object_function(self, U, V, m, X, w):
        m = tf.tile(tf.expand_dims(m, [1]), [1, tf.shape(X)[1], 1])
        U = tf.tile(tf.expand_dims(U, [1]), [1, tf.shape(X)[1], 1])
        return w * tf.reduce_sum(tf.square(U - V), axis=-1) + (1 - w) * tf.reduce_sum(tf.square(m - X), axis=-1)


    def _neg_object_function(self, U, V, m, X, w):
        m = tf.tile(tf.expand_dims(m, [1]), [1, tf.shape(X)[1], 1])
        U = tf.tile(tf.expand_dims(U, [1]), [1, tf.shape(X)[1], 1])
        return w * tf.reduce_sum(tf.square(U - V), axis=-1) + (1 - w) * tf.reduce_sum(tf.square(m - X), axis=-1)


    def loss_function(self, gamma, l2_lambda, pos_y, neg_y):

        cnt_neg = tf.shape(neg_y)[1]
        cnt_pos = tf.shape(pos_y)[1]

        pos_y = tf.reshape(tf.tile(tf.expand_dims(pos_y, -1), [1, 1, cnt_neg]), [-1, cnt_neg * cnt_pos])
        neg_y = tf.reshape(tf.tile(neg_y,[1,cnt_pos]),[-1,cnt_neg * cnt_pos])
        loss = tf.reduce_mean(tf.nn.relu(pos_y + gamma - neg_y), axis=-1)

        tv = tf.trainable_variables()
        regularization_cost = l2_lambda * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
        loss = loss + regularization_cost

        return loss

    def __mask_seq(self, input, input_length):

        mask = tf.sequence_mask(input_length, tf.shape(input)[1], dtype=tf.float32)  # [B,T]
        mask = tf.expand_dims(mask, -1)  # [B,T,1]
        mask = tf.tile(mask, [1, 1, tf.shape(input)[2]])  # [B,T,e]
        input *= mask  # [B,T,e]
        return input


    def _TE(self, t, i, d):
        if i % 2 == 0:
            return math.sin(t / math.pow(10000, 2 * i / d))
        else:
            return math.cos(t / math.pow(10000, 2 * (i - 1) / d))


    def _make_time_signal(self, size, max_timestep):
        print(max_timestep)
        te_list = []
        for t in range(int(max_timestep)):
            tmp = []
            for i in range(size):
                tmp.append(self._TE(t, i, size))
            te_list.append(tmp)
        return te_list






























