import pandas as pd
import random
import pickle
from tqdm import tqdm
from Sequential_Recommendation.make_datasets import make_datasets


# trainIterator = DataIterator('train',d_train, batch_size, neg_sample,
#                                  all_items, items_usr_clicked, shuffle=True)

class DataIterator:

    def __init__(self,
                 mode,
                 data,
                 batch_size = 128,
                 neg_sample = 1,
                 all_items = None,
                 items_usr_clicked = None,
                 shuffle = True):
        self.mode = mode
        self.data = data
        self.datasize = data.shape[0]
        self.neg_count = neg_sample
        self.batch_size = batch_size
        self.item_usr_clicked = items_usr_clicked
        self.all_items = all_items
        self.shuffle = shuffle
        self.seed = 0
        self.idx=0
        self.total_batch = round(self.datasize / float(self.batch_size))

    def __iter__(self):
        return self

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.data= self.data.sample(frac=1).reset_index(drop=True)
            self.seed = self.seed + 1
            random.seed(self.seed)

    def __next__(self):

        if self.idx  >= self.datasize:
            self.reset()
            raise StopIteration

        nums = self.batch_size
        if self.datasize - self.idx < self.batch_size:
            nums  = self.datasize - self.idx

        cur = self.data.iloc[self.idx:self.idx+nums]

        batch_user = cur['user'].values

        batch_seq = []
        for seq in cur['seq'].values:
            batch_seq.append(seq)

        batch_pos = []
        for t in cur['target'].values:
            batch_pos.append(t)

        batch_neg = []
        if self.mode == 'train':
            for u in cur['user']:
                user_item_set = set(self.all_items) - set(self.item_usr_clicked[u])
                batch_neg.append(random.sample(user_item_set,self.neg_count))

        self.idx += self.batch_size

        return (batch_user, batch_seq, batch_pos, batch_neg)

if __name__ == '__main__':

    d_train, d_test, d_info = make_datasets(5, 3, 4)
    num_usr, num_item, items_usr_clicked, _, _ = d_info
    all_items = [i for i in range(num_item)]

    # Define DataIterator

    trainIterator = DataIterator('train', d_train, 21, 5,
                                 all_items, items_usr_clicked, shuffle=True)
    for epoch in range(6):
        for data in tqdm(trainIterator,desc='epoch {}'.format(epoch),total=trainIterator.total_batch):
            batch_usr, batch_seq, batch_pos, batch_neg = data




