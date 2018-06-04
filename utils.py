import os
import nltk
import random
import collections
import numpy as np
from nltk.tokenize import WordPunctTokenizer

def load_file(folder):
    pos_folder=os.path.join(folder,'pos')
    neg_folder=os.path.join(folder,'neg')
    sentences=[]
    labels=[]
    for name in os.listdir(pos_folder):
        filename=os.path.join(pos_folder,name)
        with open(filename) as f:
            sentence=f.read()
        sentences.append(sentence)
        labels.append(0)
    for name in os.listdir(neg_folder):
        filename=os.path.join(neg_folder,name)
        with open(filename) as f:
            sentence=f.readline()
        sentences.append(sentence)
        labels.append(1)
    return np.array(sentences),np.array(labels)

def build_dict():
    word_num={}
    num_word={}
    with open('aclImdb/imdb.vocab') as f:
        for idx,line in enumerate(f):
            word_num[line[:-1]]=idx
            num_word[idx]=line[:-1]
    # sentences,_=load_file('aclImdb/imdb.vocab')
    # words=[]
    # for sentence in sentences:
    #     for word in WordPunctTokenizer().tokenize(sentence):
    #         words.append(word)
    # # print(words)
    # word_time=collections.Counter(words)
    # for idx,word in enumerate(word_time.keys()):
    #     word_num[word]=idx
    #     num_word[idx]=word
    # print(len(word_num.keys()))
    # print(idx)
    return word_num,num_word

def split_sentences(sentences):
    sentence=[]
    tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    sentence.append(tokenizer.tokenize(sentences))
    return sentence

def word2vector(sentence,word_num):
    vector=[]
    for word in WordPunctTokenizer().tokenize(sentence):
        try:
            vector.append(word_num[word])
        except:
            vector.append(0)
    return vector,len(vector)

def prepare_data(sentences,word_num):
    tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    each_sentences=[]
    for sentence in sentences:
        each_sentences.append(tokenizer.tokenize(sentence))

    vectors=[]
    word_index=[]
    for sen in each_sentences:
        each_vector=[]
        for word_sen in sen:
            vec,length=word2vector(word_sen,word_num)
            # print(length)
            if length<60:
                each_vector.append(vec)
                word_index.append(length)
            else:
                each_vector.append([0])
                word_index.append(1)
        vectors.append(each_vector)

    sen_index=[len(sen) for sen in each_sentences]
    max_sen_len=np.max(sen_index)

    max_word_len=np.max(word_index)

    # print(max_word_len)
    seq_len=np.zeros([len(sentences),max_sen_len])
    data=np.zeros((len(sentences),max_sen_len,max_word_len))
    for i,vector in enumerate(vectors):
        for j,each_vector in enumerate(vector):

            seq_len[i,j]=len(each_vector)
            data[i,j,:len(each_vector)]=each_vector

    return data,np.reshape(seq_len,-1),max_sen_len


def next_batch(batch_size,mode='train'):
    if mode=='train':
        sentences,labels=load_file('aclImdb/train')
        # print(labels)
    else:
        sentences,labels=load_file('aclImdb/test')
    length=len(sentences)
    idx=np.arange(0,length)
    np.random.shuffle(idx)
    sentences=sentences[idx]
    labels=labels[idx]
    word_num,_=build_dict()
    # print(len(sen_label))
    # for i in range(len(sen_label)):
    #     sentences,labels=sen_label[i]
    #     print(labels)
    # print(len(sentences))
    sentence_batch=[]
    label_batch=[]
    seq_len=[]
    max_len=[]
    length=len(sentences)
    start_index=0
    while(1):
        end_index = start_index + batch_size
        if end_index>=length:
            break
        sen,sen_num,sen_len=prepare_data(sentences[start_index:end_index],word_num)
        seq_len.append(sen_num)
        max_len.append(sen_len)
        sentence_batch.append(sen)
        label_batch.append(labels[start_index:end_index])
        start_index=end_index
    return sentence_batch,label_batch,seq_len,max_len

# sen,label,seq_len=next_batch(128)
# print(sen[1].shape)
# print(label)
# print(seq_len)