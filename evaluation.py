import numpy as np
import math

def Metric_PrecN(target_list, predict_list, num):

    sum = 0
    count = 0
    for i in range(len(target_list)):
        target = target_list[i]
        preds = predict_list[i]
        preds = preds[:num]
        sum += len(set(target).intersection(preds))
        count += len(preds)

    return sum / count

def Metric_RecallN(target_list, predict_list, num):

    sum = 0
    count = 0
    for i in range(len(target_list)):
        target = target_list[i]
        preds = predict_list[i]
        preds = preds[:num]
        sum += len(set(target).intersection(preds))
        count += len(target)
    return sum / count

def cal_PR(target_list,predict_list,k=[1,5,10]):

    display_list = []

    for s in k:
        prec = Metric_PrecN(target_list,predict_list,s)
        recall = Metric_RecallN(target_list,predict_list,s)
        display = "Prec@{}:{:g} Recall@{}:{:g}".format(s,round(prec,4),s,round(recall,4))
        display_list.append(display)

    return ' '.join(display_list)


def Metric_HR(TopN, target_list, predict_list):
    sums = 0
    count = 0
    for i in range(len(target_list)):
        preds = predict_list[i]
        top_preds = preds[:TopN]

        for target in target_list[i]:
            if target in top_preds:
                sums+=1
            count +=1

    return float(sums) / count

def Metric_MRR(target_list, predict_list):

    sums = 0
    count = 0
    for i in range(len(target_list)):
        preds = predict_list[i]
        for t in target_list[i]:
            rank = preds.index(t) + 1
            sums += 1 / rank
            count += 1
    return float(sums) / count

def SortItemsbyScore(item_list, item_score_list, reverse=False,remove_hist=False, usr = None, usrclick = None):

    totals = len(item_score_list)
    result_items = []
    result_score = []
    for i in range(totals):
        u = usr[i]
        u_clicks = usrclick[u]
        item_score = item_score_list[i]
        tuple_list = sorted(list(zip(item_list,item_score)),key=lambda x:x[1],reverse=reverse)

        if remove_hist:
            tmp = []
            for t in tuple_list:
                if t[0] not in u_clicks:
                    tmp.append(t)
            tuple_list = tmp

        x, y = zip(*tuple_list)
        sorted_item = list(x)
        sorted_score = list(y)
        result_items.append(sorted_item)
        result_score.append(sorted_score)

    return result_items,result_score
