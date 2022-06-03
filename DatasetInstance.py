#coding=utf-8
import os
import pickle as pkl
import numpy as np
import random
from tqdm import tqdm
from collections import OrderedDict
import unicodedata
import tensorflow as tf
from tensorflow import keras
from copy import deepcopy
class DatasetInstance(object):
    def __init__(self,fileName):
        self.DataDict=OrderedDict()
        self.UNK="unk"
        self.DataDict["text2id"]=self.construct_tokenize(fileName)
    def construct_tokenize(self,fileName):
        _,histories,response=self.create_dataset(fileName)
        text_list=[]
        for history in histories:
            text_list.extend(history)
        for res in list(response):
            text_list.extend(res)
        return self.tokenize(text_list)
    def unicode_to_ascii(self,s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    def preprocess_sentence(self,w):
        w = self.unicode_to_ascii(w.lower().strip())
        return w
    def create_dataset(self,path,is_test=False):
        word_pairs=[]
        with open(path,'r',encoding="utf-8") as f:
            for line in tqdm(f):
                line=self.preprocess_sentence(line)
                line=line.rstrip()
                contents=line.split("\t")
                if is_test:
                    label ="0"
                else:
                    label=contents[0]
                history=contents[1:-1]
                response=contents[-1:]
                word_pairs.append([label,history,response])
        return zip(*word_pairs)
    def tokenize(self,text):
        """
        text can be a list/tuple of string:["this is a demo","this a demo"]
        or 2-D list words:[["this","is","a","demo"],["this","is","a","demo"]]
        """
        Tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',lower=True,oov_token=self.UNK)
        Tokenizer.fit_on_texts(text)
        return Tokenizer
    def sequence2id(self,sequence):
        return self.DataDict["text2id"].texts_to_sequences(sequence)
    def sequences2ids(self,sequences):
        results=list()
        for seq in sequences:
            results.append(self.sequence2id(seq))
        return results
    def id2sequence(self,sequence):
        return self.DataDict["text2id"].sequences_to_texts(sequence)
    def ids2sequences(self,sequences):
        results=list()
        for seq in sequences:
            results.append(self.id2sequence(seq))
        return results
    def pad_utterance(self,utterance,config):
        """
        utterance:2-D list
        """
        tensor=keras.preprocessing.sequence.pad_sequences(utterance,maxlen=config.max_utterance_len,padding='post',truncating='post')
        return tensor
    def pad_single_instance(self,history,max_turn):
        num_turn=len(history)
        if num_turn < max_turn:#self.config.max_turn
            history=history + [[0]] *(max_turn-num_turn)
        elif num_turn > max_turn:
            history=history[num_turn-max_turn:]
        return history
    def pad_turn(self,histories,config):
        histories=[self.pad_single_instance(history,config.max_turn) for history in histories]
        histories=[self.pad_utterance(history,config) for history in histories]
        return histories
    def sample_negatives(self,response_len,sample_number):
        _index = np.random.randint(0,response_len,size=sample_number)
        return _index
    def get_visual(self,path):
        num_turn =0
        single_len = []
        with open(path,'r',encoding="utf-8") as f:
            for line in tqdm(f):
                line=self.preprocess_sentence(line)
                line=line.rstrip()
                contents=line.split("\t")
                contents = contents[1:]
                num_turn = len(contents)
                result_ids = self.sequence2id(contents)
                single_len=[len(item) for item in result_ids]
        return num_turn,single_len
    def ContraTFRecoderFeature(self,path_to_file,outfile,config,is_test=False):
        writer=tf.data.experimental.TFRecordWriter(outfile)
        def create_int_feature(values):
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return feature 
        def serialize_example(history,response,negatives):
            history=history.tolist()
            negatives=negatives.tolist()
            name_feature = {
                'history': create_int_feature(sum(history,[])),
                'response': create_int_feature(response),
                'placeholder': create_int_feature(sum(negatives,[]))}
            tf_example=tf.train.Example(features=tf.train.Features(feature=name_feature))
            return tf_example.SerializeToString()
        labels,histories,responses=self.create_dataset(path_to_file,is_test)
        container_his,container_response,container_neg=None,None,[]
        copy_res = deepcopy(list(responses))
        string_feature=[]
        for l,his,res in tqdm(zip(labels,histories,responses)):
            if l=="1":
                if len(container_neg)>0 and container_his is not None and container_response is not None:
                    sample_number = config.negatives_num - len(container_neg)
                    copy_res.remove(res)
                    sample_index = self.sample_negatives(len(copy_res),sample_number)
                    sample_negatives = [copy_res[index] for index in sample_index]
#                     sample_negatives = self.sample_negatives(copy_res,sample_number)
                    copy_res.append(res)
                    sample_negatives = [item[0] for item in sample_negatives]
                    container_neg.extend(sample_negatives)
#                     print(np.shape(container_neg))
                    #history ids
                    history_ids=self.sequence2id(container_his)
                    history_ids=self.pad_single_instance(history_ids,config.max_turn)
                    history_ids=self.pad_utterance(history_ids,config)
                    #response ids
                    response_ids=self.sequence2id(container_response)
                    response_ids=self.pad_utterance(response_ids,config)[0]
                    #negatives ids
                    negative_ids=self.sequence2id(container_neg)
                    negative_ids=self.pad_single_instance(negative_ids,config.negatives_num)
                    negative_ids=self.pad_utterance(negative_ids,config)
#                     print(negative_ids)
                    feature = (history_ids,response_ids,negative_ids)
                    string_feature.append(serialize_example(*feature)) 
                container_response=res
                container_his=his
                container_neg=list()
            elif l=='0' and not any(container_neg):
                container_neg.extend(res)
        serialized_features_dataset=tf.data.Dataset.from_tensor_slices(string_feature)
        writer.write(serialized_features_dataset)
        tf.print("save the TFRecord file {}".format(outfile))
        return len(string_feature)
    
    def SupTFRecoderFeature(self,path_to_file,outfile,config,is_test=False):
        writer=tf.data.experimental.TFRecordWriter(outfile)
        def create_int_feature(values):
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return feature 
        def serialize_example(history,response,labels):
            history=history.tolist()
            response=response.tolist()
            name_feature = {
                'history': create_int_feature(sum(history,[])),
                'response': create_int_feature(response),
                'placeholder': create_int_feature([labels])}
            tf_example=tf.train.Example(features=tf.train.Features(feature=name_feature))
            return tf_example.SerializeToString()
        labels,histories,responses=self.create_dataset(path_to_file,is_test)
        labels=[int(element) for element in labels]
        history_ids=self.sequences2ids(histories) #s * turn * seq_len
        history_ids=self.pad_turn(history_ids,config) # s * max_turn * max_seq_len
        response_ids=self.sequences2ids(responses)# s  * 1 * seq_len
        response_ids=[response[0] for response in response_ids]
        response_ids=self.pad_utterance(response_ids,config)# s* seq_len
        string_feature=[]
        for feature in zip(history_ids,response_ids,labels):
            string_feature.append(serialize_example(*feature))
        serialized_features_dataset=tf.data.Dataset.from_tensor_slices(string_feature)
        writer.write(serialized_features_dataset)
        tf.print("save the TFRecord file {}".format(outfile))
        return len(string_feature)
    def generate_embedding(self,config):
        word_index=self.DataDict["text2id"].word_index
        vocab_size=max(word_index.values())+1
        embedding_matrix = np.random.random((vocab_size, config.emb_size))
        embedding_matrix[0]=np.zeros(config.emb_size)
        pre_train_emb=os.path.join(config.emb_path,config.corpus,"vectors.txt")
        with open(pre_train_emb,"r",encoding="utf-8") as f:
            for line in tqdm(f):
                split_element=line.split()
                if config.lower_case:
                    token=split_element[0].lower()
                else:
                    token=split_element[0]
                vector=split_element[-config.emb_size:]
                assert len(vector) == config.emb_size
                if token in word_index:
                    embedding_matrix[word_index[token]]=np.asarray([float(item) for item in vector],dtype=np.float32)
        emb_file=os.path.join(config.data_dir,config.corpus,config.emb_file)
        with open(emb_file,"wb") as f:
            pkl.dump(embedding_matrix,f)
        tf.print("save the emb file {}".format(emb_file))
    def Contra_batch_data(self,config,recordFile,batch_size,is_training=False):
        feature_description = {
            'history': tf.io.FixedLenFeature([config.max_turn * config.max_utterance_len], tf.int64),
            'response': tf.io.FixedLenFeature([config.max_utterance_len], tf.int64),
            'placeholder': tf.io.FixedLenFeature([config.negatives_num * config.max_utterance_len], tf.int64)}
        def _parse_function(example):
            example= tf.io.parse_single_example(example,feature_description)
            for name in list(example.keys()):
                t=example[name]
                if t.dtype==tf.int64:
                    t=tf.cast(t,tf.int32)
                example[name]=t
            return example
        d=tf.data.TFRecordDataset(recordFile)
        if is_training:
            d=d.repeat(config.num_train_epochs)
            d=d.shuffle(buffer_size=2048)
        parse_data=d.map(_parse_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        parse_data = parse_data.prefetch(tf.data.experimental.AUTOTUNE).batch(batch_size,drop_remainder=config.drop_last)
        return parse_data
            
    def Sup_batch_data(self,config,recordFile,batch_size,is_training=False):
        feature_description = {
            'history': tf.io.FixedLenFeature([config.max_turn * config.max_utterance_len], tf.int64),
            'response': tf.io.FixedLenFeature([config.max_utterance_len], tf.int64),
            'placeholder': tf.io.FixedLenFeature([],tf.int64)}
        def _parse_function(example):
            example= tf.io.parse_single_example(example,feature_description)
            for name in list(example.keys()):
                t=example[name]
                if t.dtype==tf.int64:
                    t=tf.cast(t,tf.int32)
                example[name]=t
            return example
        d=tf.data.TFRecordDataset(recordFile)
        if is_training:
            d=d.repeat(config.num_train_epochs)
            d=d.shuffle(buffer_size=2048)
        parse_data=d.map(_parse_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        parse_data = parse_data.prefetch(tf.data.experimental.AUTOTUNE).batch(batch_size,drop_remainder=config.drop_last)
        return parse_data

    