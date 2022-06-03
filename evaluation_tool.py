#coding=utf-8
import numpy as np
import json
from collections import OrderedDict
#MAP
def mean_average_precision(sort_data):
    #to do
    count_1 = 0
    sum_precision = 0
    for index in range(len(sort_data)):
        if sort_data[index][1] == 1:
            count_1 += 1
            sum_precision += 1.0 * count_1 / (index+1)
    return sum_precision / count_1
#MRR
def mean_reciprocal_rank(sort_data):
    sort_lable = [s_d[1] for s_d in sort_data]
    assert 1 in sort_lable
    return 1.0 / (1 + sort_lable.index(1))
#P@1
def precision_at_position_1(sort_data):
    if sort_data[0][1] == 1:
        return 1
    else:
        return 0
#R10@k
def recall_at_position_k_in_10(sort_data, k):
    sort_lable = [s_d[1] for s_d in sort_data]
    select_lable = sort_lable[:k]
    return 1.0 * select_lable.count(1) / sort_lable.count(1)
# douban evaluation metrics
def evaluation_one_session(data):
    sort_data = sorted(data, key=lambda x: x[0], reverse=True)
    m_a_p = mean_average_precision(sort_data)
    m_r_r = mean_reciprocal_rank(sort_data)
    p_1 = precision_at_position_1(sort_data)
    r_1 = recall_at_position_k_in_10(sort_data, 1)
    r_2 = recall_at_position_k_in_10(sort_data, 2)
    r_5 = recall_at_position_k_in_10(sort_data, 5)
    return m_a_p, m_r_r, p_1, r_1, r_2, r_5

def DoubanMetrics(scores,labels,count = 10):
    eval_metrics=OrderedDict()
    R1,R2,R5,MRR,MAP,P1= 0,0,0,0,0,0
    total = 0
    assert len(scores.shape) ==2
    scores = scores[:,1].tolist()
    assert len(scores)==len(labels)
    for i in range(0,len(scores),count):
        data=[]
        g_score=scores[i:i+count]
        g_label=labels[i:i+count]
        for score,label in zip(g_score,g_label):
            data.append((score,label))
        if 1 in g_label:
            total = total+1
            _map,mrr,p1,r1, r2, r5=evaluation_one_session(data)
            MAP +=_map
            MRR +=mrr
            R1 +=r1
            R2 +=r2
            R5 +=r5
            P1 +=p1
    eval_metrics["R10@1"]=R1*1.0 / total
    eval_metrics["R10@2"]=R2*1.0 / total
    eval_metrics["R10@5"]=R5*1.0 / total
    eval_metrics["P@1"]=P1*1.0 / total
    eval_metrics["MRR"]=MRR*1.0 / total
    eval_metrics["MAP"]=MAP*1.0 / total
    return eval_metrics
def evaluation_mutual_session(data):
    sort_data = sorted(data, key=lambda x: x[0], reverse=True)
    m_a_p = mean_average_precision(sort_data)
    m_r_r = mean_reciprocal_rank(sort_data)
    p_1 = precision_at_position_1(sort_data)
    r_1 = recall_at_position_k_in_10(sort_data, 1)
    r_2 = recall_at_position_k_in_10(sort_data, 2)
    r_4 = recall_at_position_k_in_10(sort_data, 4)
    return m_a_p, m_r_r, p_1, r_1, r_2, r_4

def MutualMetrics(scores,labels,count = 4):
    eval_metrics=OrderedDict()
    R1,R2,R5,MRR,MAP,P1= 0,0,0,0,0,0
    total = 0
    assert len(scores.shape) ==2
    scores = scores[:,1].tolist()
    assert len(scores)==len(labels)
    save_prediction(scores, count)
    for i in range(0,len(scores),count):
        data=[]
        g_score=scores[i:i+count]
        g_label=labels[i:i+count]
        for score,label in zip(g_score,g_label):
            data.append((score,label))
        if 1 in g_label:
            total = total+1
            _map,mrr,p1,r1, r2, r5=evaluation_mutual_session(data)
            MAP +=_map
            MRR +=mrr
            R1 +=r1
            R2 +=r2
            R5 +=r5
            P1 +=p1
    eval_metrics["R10@1"]=R1*1.0 / total
    eval_metrics["R10@2"]=R2*1.0 / total
    eval_metrics["R10@5"]=R5*1.0 / total
    eval_metrics["P@1"]=P1*1.0 / total
    eval_metrics["MRR"]=MRR*1.0 / total
    eval_metrics["MAP"]=MAP*1.0 / total
    return eval_metrics
def TopMetrics(top_list,label_list,logits_list):
    eval_metrics=dict()
    for num in top_list:
        key_metrics = "acc_%d"%(num)
        eval_metrics[key_metrics] = top_N(label_list, logits_list, num)
    return eval_metrics

def ContraDoubanMetrics(scores_list,labels_list,count=10):
    eval_metrics=OrderedDict()
    R1,R2,R5,MRR,MAP,P1= 0,0,0,0,0,0
    total = 0
    assert len(scores_list)==len(labels_list)
    for i in range(0,len(scores_list),count):
        data=[]
        g_score=scores_list[i:i+count]
        g_label=labels_list[i:i+count]
        for score,label in zip(g_score,g_label):
            data.append((score,label))
        if 1 in g_label:
            total = total+1
            _map,mrr,p1,r1, r2, r5=evaluation_one_session(data)
            MAP +=_map
            MRR +=mrr
            R1 +=r1
            R2 +=r2
            R5 +=r5
            P1 +=p1
    eval_metrics["R10@1"]=R1*1.0 / total
    eval_metrics["R10@2"]=R2*1.0 / total
    eval_metrics["R10@5"]=R5*1.0 / total
    eval_metrics["P@1"]=P1*1.0 / total
    eval_metrics["MRR"]=MRR*1.0 / total
    eval_metrics["MAP"]=MAP*1.0 / total
    return eval_metrics
def save_prediction(scores_list,count = 4):
    tests_id =[]
    with open("datasets/mutual/test_id.txt","r",encoding="utf-8") as f:
        for line in f:
            tests_id.append(line.rstrip())
    rank_result = []
    labels_map={0:"A",1:"B",2:"C",3:"D"}
    for i in range(0,len(scores_list),count):
        g_score=scores_list[i:i+count]
        temp_dict = OrderedDict()
        for i,score in enumerate(g_score):
            temp_dict[labels_map[i]] = score
        results = dict(sorted(temp_dict.items(), key = lambda kv:(kv[1], kv[0]),reverse=True))
        rank_result.append(list(results.keys()))
    predict_results =OrderedDict()
    for t_id,rank_value in zip(tests_id,rank_result):
        predict_results[t_id] = rank_value
    with open("TrainModel/mutual_predictions.txt",'w',encoding="utf-8") as pred_writer:
        for _key,_value in predict_results.items():
            pred_writer.write(_key+"\t"+"\t".join(_value)+"\n")
        
        
def ContraMutualMetrics(scores_list,labels_list,count = 4):
    eval_metrics=OrderedDict()
    R1,R2,R5,MRR,MAP,P1= 0,0,0,0,0,0
    total = 0
    assert len(scores_list)==len(labels_list)
    for i in range(0,len(scores_list),count):
        data=[]
        g_score=scores_list[i:i+count]
        g_label=labels_list[i:i+count]
        for score,label in zip(g_score,g_label):
            data.append((score,label))
        if 1 in g_label:
            total = total+1
            _map,mrr,p1,r1, r2, r5=evaluation_mutual_session(data)
            MAP +=_map
            MRR +=mrr
            R1 +=r1
            R2 +=r2
            R5 +=r5
            P1 +=p1
    eval_metrics["R10@1"]=R1*1.0 / total
    eval_metrics["R10@2"]=R2*1.0 / total
    eval_metrics["R10@5"]=R5*1.0 / total
    eval_metrics["P@1"]=P1*1.0 / total
    eval_metrics["MRR"]=MRR*1.0 / total
    eval_metrics["MAP"]=MAP*1.0 / total
    return eval_metrics
            
def top_N(labels,logits,n=1):
    """
    parameters:
        labels:N
        logits:N * (m+1)
    """
    scores_dict=OrderedDict()
    for _batch_index,scores in enumerate(logits):
        sample_dict=dict()
        for _index,score in enumerate(scores):
            sample_dict[_index]=score
        sample_dict=sorted(sample_dict.items(),key = lambda x:x[1],reverse = True)
        results=[tuple_item[0] for tuple_item in sample_dict]
        scores_dict[_batch_index] = results
    all_scores = 0
    for _key,_value in scores_dict.items():
        n_value=_value[:n]
        if 0 in n_value:
            all_scores +=1
    acc_n = all_scores *1.0 / len(labels)
    return acc_n
        
    
    



