import tensorflow as tf
import math


class TransRec(object):


    def __init__(self, emb_size, num_usr, num_item):

        self.emb_size = emb_size
        self.item_count = num_item
        self.user_count = num_usr

        self.init = self.init = tf.random_uniform_initializer(minval= -6 / math.sqrt(self.emb_size),maxval= 6 / math.sqrt(self.emb_size))

        self.input_Seq = tf.placeholder(tf.int32, [None,1]) #[B,T]
        self.input_Usr = tf.placeholder(tf.int32, [None]) #[B]
        self.input_NegT = tf.placeholder(tf.int32, [None, None]) #[B,F]
        self.input_PosT = tf.placeholder(tf.int32, [None,1]) #[B]

        self.loss, self.output = self.build_model(self.input_Seq,self.input_Usr,self.input_NegT,self.input_PosT)


    def loss_function(self, X_uti, X_utj):
        return - 1* tf.reduce_mean(tf.log(tf.sigmoid(tf.squeeze(X_uti - X_utj))),-1)

    def l2_distance(self,x,y):
        a = tf.reduce_sum(tf.square(x - y), axis=-1,keep_dims=True)
        return a

    def build_model(self, in_Seq, in_Usr, in_Neg, in_Pos):

        self.user_emb = tf.get_variable("user_emb", [self.user_count, self.emb_size],initializer=self.init)
        self.item_emb = tf.get_variable("item_emb", [self.item_count, self.emb_size], initializer=self.init)

        self.Beta = tf.get_variable("Beta",[self.item_count,1], initializer=self.init)
        self.T = T = tf.get_variable("T", [self.emb_size], initializer=self.init)

        last_item = tf.nn.embedding_lookup(self.item_emb,in_Seq) #[B,1,e]
        next_item = tf.nn.embedding_lookup(self.item_emb,in_Pos) #[B,1,e]
        neg_items = tf.nn.embedding_lookup(self.item_emb,in_Neg) #[B,n,e]
        tu = tf.expand_dims(tf.nn.embedding_lookup(self.user_emb,in_Usr),1)#[B,1,e]

        last_item = tf.clip_by_norm(last_item, 1, -1)
        next_item = tf.clip_by_norm(next_item, 1, -1)
        neg_items = tf.clip_by_norm(neg_items, 1, -1)
        tu = tf.clip_by_norm(tu, 1, -1)

        output = tu + T + last_item
        # TransRec
        bias_pos = tf.nn.embedding_lookup(self.Beta, in_Pos)  # [B,1,1]
        pos_score = bias_pos - self.l2_distance(output, next_item)
        bias_neg = tf.nn.embedding_lookup(self.Beta, in_Neg)
        neg_score = bias_neg - self.l2_distance(output, neg_items)
        loss = self.loss_function(pos_score,neg_score)

        return loss,output




    def predict(self):

        tu = tf.expand_dims(tf.nn.embedding_lookup(self.user_emb, self.input_Usr), 1)

        last_item = tf.nn.embedding_lookup(self.item_emb, self.input_Seq) #[B,T,1]

        all_index = tf.convert_to_tensor([[i for i in range(self.item_count)]])
        all_index = tf.tile(all_index, [tf.shape(self.input_Usr)[0], 1])

        next_item = tf.nn.embedding_lookup(self.item_emb,all_index)
        last_item = tf.clip_by_norm(last_item, 1, -1)
        next_item = tf.clip_by_norm(next_item, 1, -1)
        tu = tf.clip_by_norm(tu, 1, -1)

        Beta = tf.nn.embedding_lookup(self.Beta,all_index)

        score = Beta - self.l2_distance(tu + self.T + last_item, next_item)

        return score

