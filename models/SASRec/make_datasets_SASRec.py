import pandas as pd
import numpy as np

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def make_datasets(data, len_Seq = 0, len_Tag =0, len_Pred = 0):

    #file_path = 'input/u.data'

    p = data.groupby('item')['user'].count().reset_index().rename(columns={'user':'item_count'})
    data = pd.merge(data,p,how='left',on='item')
    data = data[data['item_count'] > 5].drop(['item_count'],axis=1)

    # ReMap item ids
    item_unique = data['item'].unique().tolist()
    item_map = dict(zip(item_unique, range(1,len(item_unique) + 1)))
    item_map[0] = 0
    all_item_count = len(item_map)
    data['item'] = data['item'].apply(lambda x: item_map[x])

    # ReMap usr ids
    user_unique = data['user'].unique().tolist()
    user_map = dict(zip(user_unique, range(1, len(user_unique) + 1)))
    user_map[0] = 0
    all_user_count = len(item_map)
    data['user'] = data['user'].apply(lambda x: user_map[x])

    # Get user session
    data = data.sort_values(by=['user','timestamps']).reset_index(drop=True)

    # 生成用户序列
    user_sessions = data.groupby('user')['item'].apply(lambda x: x.tolist()) \
        .reset_index().rename(columns={'item': 'item_list'})

    train_users = []
    train_seqs = []
    train_targets = []

    test_users = []
    test_seqs = []
    test_targets = []

    items_usr_clicked = {}

    for index, row in user_sessions.iterrows():
        user = row['user']
        items = row['item_list']

        items_usr_clicked[user] = items

        user_train = items[:-1]

        train_seq = np.zeros([len_Seq], dtype=np.int32)
        train_pos = np.zeros([len_Seq], dtype=np.int32)

        nxt = user_train[-1]
        idx = len_Seq - 1
        for i in reversed(user_train[:-1]):
            train_seq[idx] = i
            train_pos[idx] = nxt
            nxt = i
            idx -= 1
            if idx == -1: break
        train_users.append(user)
        train_seqs.append(train_seq)
        train_targets.append(train_pos)

        test_seq = np.zeros([len_Seq], dtype=np.int32)


        user_test = items
        nxt = user_test[-1]
        test_pos = [nxt]

        idx = len_Seq - 1
        for i in reversed(user_test[:-1]):
            test_seq[idx] = i
            idx -= 1
            if idx == -1: break
        test_users.append(user)
        test_seqs.append(test_seq)
        test_targets.append(test_pos)






    d_train = pd.DataFrame({'user':train_users,'seq':train_seqs,'target':train_targets})

    d_test = pd.DataFrame({'user': test_users, 'seq': test_seqs, 'target': test_targets})

    d_info= (all_user_count, all_item_count, items_usr_clicked, user_map, item_map)

    return d_train,d_test,d_info


if __name__ == '__main__':
    make_datasets(5,3,2)



'''

src data

196	242	3	881250949
186	302	3	891717742
22	377	1	878887116
244	51	2	880606923
166	346	1	886397596
298	474	4	884182806
115	265	2	881171488
253	465	5	891628467
305	451	3	886324817
6	86	3	883603013
62	257	2	879372434
286	1014	5	879781125
200	222	5	876042340
210	40	3	891035994
224	29	3	888104457



train_data
                          seq            target  user
0     [1, 290, 492, 381, 752]    [467, 523, 11]     1
1   [290, 492, 381, 752, 467]    [523, 11, 673]     1
2   [492, 381, 752, 467, 523]   [11, 673, 1046]     1
3    [381, 752, 467, 523, 11]  [673, 1046, 650]     1
4    [752, 467, 523, 11, 673]  [1046, 650, 378]     1
5   [467, 523, 11, 673, 1046]   [650, 378, 180]     1
6   [523, 11, 673, 1046, 650]   [378, 180, 390]     1
7   [11, 673, 1046, 650, 378]   [180, 390, 666]     1
8  [673, 1046, 650, 378, 180]   [390, 666, 513]     1
9  [1046, 650, 378, 180, 390]   [666, 513, 432]     1

test_data
                            seq       target  user
0    [633, 657, 1007, 948, 364]  [522, 0, 0]     1
1       [26, 247, 49, 531, 146]   [32, 0, 0]     2
2      [459, 477, 369, 770, 15]  [306, 0, 0]     3
3  [1093, 946, 1101, 690, 1211]  [526, 0, 0]     4
4     [732, 266, 669, 188, 253]  [986, 0, 0]     5
5      [410, 446, 104, 782, 96]   [26, 0, 0]     6
6     [146, 817, 536, 694, 186]  [525, 0, 0]     7
7      [395, 669, 281, 289, 98]  [731, 0, 0]     8
8     [588, 671, 369, 292, 304]  [250, 0, 0]     9
9        [472, 222, 82, 716, 8]  [131, 0, 0]    10


'''












