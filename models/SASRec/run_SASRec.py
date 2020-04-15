import sys
import os
from tqdm import tqdm
import pandas as pd
import argparse
sys.path.append("..")

os.environ["CUDA_VISIBLE_DEVICES"]='1'
import tensorflow as tf
import numpy as np

from model_SASRec import SASRec
from make_datasets_SASRec import make_datasets
from DataInput_SASRec import DataIterator
from evaluation import SortItemsbyScore,Metric_HR,Metric_MRR


def parse_args():
    parser = argparse.ArgumentParser(description='SASRec')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--display_step', type=int, default=256)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=3e-3)
    parser.add_argument('--num_blocks', type=float, default=2)
    parser.add_argument('--num_heads', type=float, default=1)
    parser.add_argument('--keep_prob', type=float, default=0.8)
    parser.add_argument('--l2_lambda', type=float, default=1e-6)
    return parser.parse_args()




if __name__ == '__main__':

    # Get Params
    args = parse_args()

    # make datasets

    print('==> make datasets <==')
    file_path = '../../datasets/ratings1m.dat'
    names = ['user', 'item', 'rateing', 'timestamps']
    data = pd.read_csv(file_path, header=None, sep='::', names=names)
    d_train, d_test, d_info = make_datasets(data, args.max_len)


    num_usr, num_item, items_usr_clicked, _, _ = d_info
    all_items = [i for i in range(num_item)]

    # Define DataIterator
    trainIterator = DataIterator('train',d_train, args.batch_size, args.max_len,
                                 all_items, items_usr_clicked, shuffle=True)
    testIterator = DataIterator('test',d_test, args.batch_size,  shuffle=False)

    # Define Model

    model = SASRec(usernum=num_user,
                   itemnum=num_item,
                   emb_size=args.emb_size,
                   max_Seqlens=args.max_len,
                   num_blocks=args.num_blocks,
                   num_heads=args.num_heads,
                   dropout_rate=args.keep_prob)


    #model = SASRec(num_usr,num_item,args)
    score_pred = model.predict(all_items)
    loss = model.loss

    # Define Optimizer
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss,global_step=global_step)

    # Training and test for every epoch
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.num_epochs):

            #train
            cost_list = []
            for train_input in tqdm(trainIterator,desc='epoch {}'.format(epoch),total=trainIterator.total_batch):
                batch_usr, batch_seq, batch_pos, batch_neg = train_input
                feed_dict = {model.u: batch_usr, model.input_seq: batch_seq,
                            model.pos: batch_pos, model.neg: batch_neg,
                            model.is_training :True}
                _, step, cost= sess.run([train_op, global_step, loss],feed_dict)
                cost_list.append(cost)
            mean_cost = np.mean(cost_list)
            #saver.save(sess, FLAGS.save_path)

            # test
            pred_list = []
            next_list = []
            user_list = []

            if epoch % 10 != 0:
                continue

            for test_input in testIterator:
                batch_usr, batch_seq, batch_pos, batch_neg = test_input
                feed_dict = {model.u: batch_usr, model.input_seq: batch_seq, model.is_training: False}
                pred = sess.run(score_pred, feed_dict)  # , options=options, run_metadata=run_metadata)

                pred_list += pred.tolist()
                next_list += list(batch_pos)
                user_list += list(batch_usr)


            sorted_items,sorted_score = SortItemsbyScore(all_items,pred_list,remove_hist=True
            ,reverse = True,usr=user_list,usrclick=items_usr_clicked)


            hr50 = Metric_HR(50, next_list,sorted_items)
            Mrr = Metric_MRR(50,next_list,sorted_items)

            print(" epoch {}, mean_loss{:g}, test HR@50: {:g} MRR: {:g}"
                .format(epoch + 1, mean_cost, hr50, Mrr))









