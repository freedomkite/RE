#encoding:utf-8

from collections import Counter
import numpy as np
import random


#读取文件
def readFile(file):
    buff=[]
    with open(file,'r') as f_r:
        for line in f_r:
            line=line.strip().decode('utf-8').split()
            buff.append(line)
    return buff
#读取数据
'''src1存储句子，src2存储词性，src3存储位置，src4存储标签'''
def load_data(src1,src3,src4):
    sentences=readFile(src1)
    pos=readFile(src2)
    loc=readFile(src3)
    label=[]
    with open(src4,'r') as f_r4:
        for line in f_r4:
            line=line.strip().decode('utf-8')
            label.append(line)
    return sentences,pos,loc,label
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
	
#构建问题类别字典
def build_label(label):
    label_dict={}
    for la in label:
        if la not in label_dict:
            label_dict[la]=len(label_dict)
        else:
            pass
    return label_dict
	
#将句子向量化
'''将句子向量化编号的矩阵'''
def vectorize(data, word_dict,pos_dict,loc_dict,label_dict, max_len):
    print "len(data)"
    sentences,poss,locs,labels=data
    num_data = len(sentences)
    # index = [i for i in range(len(sentences))]
    # random.shuffle(index)
    # sentences=[sentences[i] for i in index]
    # pos=[pos[i] for i in index]
    # labels=[labels[i] for i in index]

    sents_vec = np.zeros((num_data, max_len), dtype=int)
    pos_vec = np.zeros((num_data, max_len), dtype=int)
    loc_vec = np.zeros((num_data, max_len), dtype=int)
    
    y_vec=np.zeros((num_data, len(label_dict)), dtype=int)
    for idx, (sent, pos,loc,label) in enumerate(zip(sentences, poss,locs,labels)):
        #此处主要是处理测试语料中可能出现的长句子，长句子进行截断
        if len(sent)>max_len:
            sent=sent[:max_len]
        vec = [word_dict[w] if w in word_dict else 0 for w in sent]
        sents_vec[idx, :len(vec)] = vec
        if len(pos)>max_len:
            pos=pos[:max_len]
        vec1 = [pos_dict[p] if p in pos_dict else 0 for p in pos]
        pos_vec[idx, :len(vec1)] = vec1 
        if len(loc)>max_len:
            loc=loc[:max_len]
        vec2 = [loc_dict[lo] if lo in loc_dict else 0 for lo in loc]
        loc_vec[idx, :len(vec2)] = vec2
        
        #print label
        # y = [1 if i==label_dict[label] else 0 for i in range(len(label_dict))]
        # y_vec[idx]=y
        y = [label_dict[l] if l in label_dict else 0 for l in labels]
    return sents_vec,pos_vec,loc_vec,y#y_vec
	
   




