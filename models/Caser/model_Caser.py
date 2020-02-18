import tensorflow as tf
import math


class Caser(object):


    def __init__(self, emb_size, num_usr, num_item, len_Seq, len_Tag, h_size = 16, v_size = 4, l2_lambda = 0.0):

        self.emb_size = emb_size
        self.item_count = num_item
        self.user_count = num_usr
        self.v_size = v_size
        self.h_size = h_size
        self.l2_lambda = l2_lambda
        self.len_Seq = len_Seq
        self.len_Tag = len_Tag,


        self.u_init = tf.keras.initializers.he_normal()
        self.init = tf.random_normal_initializer(0, 0.5 / self.emb_size)

        self.input_Seq = tf.placeholder(tf.int32, [None,self.len_Seq]) #[B,T]
        self.input_Usr = tf.placeholder(tf.int32, [None]) #[B]
        self.input_NegT = tf.placeholder(tf.int32, [None, None]) #[B,F]
        self.input_PosT = tf.placeholder(tf.int32, [None,None]) #[B]

        self.input_keepprob = tf.placeholder(tf.float32, name='keep_prob')

        self.loss, self.output = self.build_model(self.input_Seq,self.input_Usr
                                             ,self.input_NegT,self.input_PosT,self.input_keepprob)


    def build_model(self,in_Seq, in_Usr, in_Neg, in_Pos, in_KP):
        with tf.variable_scope('Caser'):

            self.item_emb = tf.get_variable("item_emb", [self.item_count, self.emb_size]) #[N,e]
            self.user_emb = tf.get_variable("user_emb", [self.user_count, self.emb_size]) #[N,e]

            self.W = tf.get_variable('W',[self.item_count,2*self.emb_size])
            self.b = tf.get_variable('b',[self.item_count,1])

            usr_seq = tf.nn.embedding_lookup(self.item_emb, in_Seq)  # [B,T,e]
            Pu = tf.nn.embedding_lookup(self.user_emb, in_Usr)

            convs = []

            # Horizontal convolutional layer
            for kernal in range(1,self.len_Seq + 1):
                hconv = tf.layers.conv1d(usr_seq ,self.h_size, kernal)
                hconv = tf.nn.relu(hconv)
                max_hconv = tf.reduce_max(hconv, axis=-2)
                convs.append(max_hconv)

            # Vertical convolutional layer
            u = tf.transpose(usr_seq, [0, 2, 1])
            vconv = tf.layers.conv1d(u, self.v_size, 1)
            vconv = tf.nn.relu(vconv)
            vconv = tf.layers.flatten(vconv)
            convs.append(vconv)

            # Concatenate the outputs of the two convolutional layers
            s = tf.concat(convs, axis=-1)
            s = tf.nn.dropout(s,keep_prob=in_KP)
            z = tf.layers.dense(s,units=self.emb_size,use_bias=True)
            z = tf.concat([z, Pu], axis=-1)



            pos_items = tf.nn.embedding_lookup(self.W, in_Pos)
            pos_b = tf.nn.embedding_lookup(self.b, in_Pos)
            posy = tf.matmul(tf.expand_dims(z,1),tf.transpose(pos_items,[0,2,1]))
            posy = tf.sigmoid(tf.squeeze(posy) + tf.squeeze(pos_b))

            neg_items = tf.nn.embedding_lookup(self.W, in_Neg)
            neg_b = tf.nn.embedding_lookup(self.b, in_Neg)
            negy = tf.matmul(tf.expand_dims(z, 1), tf.transpose(neg_items, [0, 2, 1]))
            negy = tf.sigmoid(tf.squeeze(negy) + tf.squeeze(neg_b))

            positive_loss = -1 * tf.reduce_mean(tf.log(posy),axis=-1)
            negative_loss = -1 * tf.reduce_mean(tf.log(1 - negy),axis=-1)

            tv = tf.trainable_variables()
            l2_loss = self.l2_lambda * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

            loss = positive_loss + negative_loss + l2_loss

            return loss ,z

    def predict(self):

        score = tf.matmul(self.output, tf.transpose(self.W, [1,0]))
        sb = tf.expand_dims(tf.squeeze(self.b),[0])
        score =  tf.sigmoid(score + sb)
        print(score.get_shape())
        return score

























