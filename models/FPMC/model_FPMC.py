import tensorflow as tf
import math


class FPMC(object):


    def __init__(self, emb_size, num_usr, num_item, len_Seq, len_Tag):

        self.emb_size = emb_size
        self.item_count = num_item
        self.user_count = num_usr
        self.len_Seq = len_Seq
        self.len_Tag = len_Tag,

        self.init = tf.random_normal_initializer(0,0.1)


        self.input_Seq = tf.placeholder(tf.int32, [None,self.len_Seq]) #[B,T]
        self.input_Usr = tf.placeholder(tf.int32, [None]) #[B]
        self.input_NegT = tf.placeholder(tf.int32, [None, None]) #[B,F]
        self.input_PosT = tf.placeholder(tf.int32, [None,None]) #[B]


        self.loss = self.build_model(self.input_Seq,self.input_Usr,self.input_NegT,self.input_PosT)


    def PMFC(self,Vui,Viu,Vil,Vli):
        '''

        :param Vui: [b,1,e]
        :param Viu: [b,S,e]
        :param Vil: [b,S,e]
        :param Vli: [b,L,e]
        :return:
        '''

        # MF
        mf = tf.matmul(Vui,tf.transpose(Viu,[0,2,1])) # [b,1,S]
        mf = tf.squeeze(mf,1) # [b, S]

        #PMF
        pmf = tf.matmul(Vil,tf.transpose(Vli,[0,2,1])) #[b,S,L]
        pmf = tf.reduce_mean(pmf,-1) #[b,S,1]
        x = pmf + mf #[B,S]

        return  x

    def loss_function(self, X_uti, X_utj):
        return - 1* tf.reduce_mean(tf.log(tf.sigmoid(tf.squeeze(X_uti - X_utj))),-1)

    def build_model(self, in_Seq, in_Usr, in_Neg, in_Pos):

        self.UI_emb = tf.get_variable("UI_emb", [self.user_count, self.emb_size],initializer=self.init)  #[N,e]
        self.IU_emb = tf.get_variable("IU_emb", [self.item_count, self.emb_size],initializer=self.init)  # [N,e]
        self.LI_emb = tf.get_variable("LI_emb", [self.item_count, self.emb_size],initializer=self.init)  #[N,e]
        self.IL_emb = tf.get_variable("IL_emb", [self.item_count, self.emb_size],initializer=self.init)  # [N,e]

        ui = tf.nn.embedding_lookup(self.UI_emb,in_Usr) #[b,1,1]
        ui = tf.expand_dims(ui,1)
        seq = tf.nn.embedding_lookup(self.LI_emb,in_Seq) #[b,l,1]

        pos_iu = tf.nn.embedding_lookup(self.IU_emb,in_Pos) #[b,1,1]
        pos_il = tf.nn.embedding_lookup(self.IL_emb, in_Pos)#[b,1,1]
        pos_score = self.PMFC(ui,pos_iu,pos_il,seq)

        neg_iu = tf.nn.embedding_lookup(self.IU_emb, in_Neg)
        neg_il = tf.nn.embedding_lookup(self.IL_emb, in_Neg)
        neg_score = self.PMFC(ui, neg_iu, neg_il, seq)


        loss = self.loss_function(pos_score,neg_score)

        return loss




    def predict(self):

        ui = tf.nn.embedding_lookup(self.UI_emb, self.input_Usr)  #[B,1,1]
        ui = tf.expand_dims(ui, 1)
        seq = tf.nn.embedding_lookup(self.LI_emb, self.input_Seq) #[B,T,1]
        pos_iu = tf.tile(tf.expand_dims(self.IU_emb,0),[tf.shape(self.input_Usr)[0] ,1,1])
        pos_il = tf.tile(tf.expand_dims(self.IL_emb, 0),[tf.shape(self.input_Usr)[0], 1, 1])

        score = self.PMFC(ui, pos_iu, pos_il, seq)

        return score

























