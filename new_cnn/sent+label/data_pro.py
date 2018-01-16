#encoding:utf-8

from collections import Counter
import numpy as np
import random


#读取数据
def load_data(file):
    sentences=[]
	labels=[]
    with open(file,'r') as f_r:
        for line in f_r:
            line=line.strip().decode('utf-8').split()
            if line:
				tmp=line.split()
				sentences.append(tmp[1:])
				labels.append(tmp[0])
    return sentences,labels

#构建字典，key为词语，value为编号
def build_dict(sentences):
    word_count = Counter()
    max_len=0
    word_count['unk']+=1
    for sent in sentences:
        if len(sent)>max_len:
            max_len=len(sent)
        for w in sent:
            word_count[w]+=1
    ls=word_count.most_common()
    word_dict={w[0]:index for (index,w) in enumerate(ls)}
    return word_dict,max_len
	
#构建标签字典
def build_label(labels):
    label_dict={}
    for la in labels:
        if la not in label_dict:
            label_dict[la]=len(label_dict)
        else:
            pass
    return label_dict
	
#将句子向量化

def vectorize(data, word_dict,label_dict, max_len):
    sentences,labels=data
    num_data = len(sentences)
	
    sents_vec = np.zeros((num_data, max_len), dtype=int)
    y_vec=np.zeros((num_data, len(label_dict)), dtype=int)
    for idx, (sent, label) in enumerate(zip(sentences,labels)):
        if len(sent)>max_len:
            sent=sent[:max_len]
        vec = [word_dict[w] if w in word_dict else 0 for w in sent]
        sents_vec[idx, :len(vec)] = vec
        y = [label_dict[l] if l in label_dict else 0 for l in labels]
    return sents_vec,y
	
   




