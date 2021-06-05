import re
import pandas as pd
import os
import sys
import math

sys.path.append('../')
cur_path = os.path.dirname(__file__)

programa = 'L6N_20160123'
enfoque = 'distintivo'

name_enfoque = 'DIST' if enfoque == 'distintivo' else 'AGLO'
path_to_file = '../datasets/' + programa + '/' + enfoque + '/' + programa + '-ORIG.txt'

dataset_file = os.path.relpath(path_to_file, cur_path)

def get_hashtags(programa):
    if programa =='L6N_20151024':
        hashtags = ['#L6Nalsina', '#L6Nlaicismo', '#L6Nprotestas', '#L6Nprecioluz']
        return hashtags
    elif programa =='L6N_20151031':
        hashtags = ['#L6Nrivera24h', '#L6Nencampaña', '#L6Nbalance']
        return hashtags
    elif programa =='L6N_20151107':
        hashtags = ['#L6Nretocatalán', '#L6Npizarrasparo']
        return hashtags
    elif programa =='L6N_20151121':
        hashtags = ['#L6Nspielberg', '#L6NfranKO', '#L6Nayudasocial']
        return hashtags
    elif programa =='L6N_20151128':
        hashtags = ['#L6Nantiyihadismo', '#L6Nsueldos', '#L6Nmarujatorres', '#L6Nnoalaviolencia', '#L6Nnomanipulación']
        return hashtags
    elif programa =='L6N_20151205':
        hashtags = ['#L6Nretoempleo', '#L6Npuertagiratoria', '#L6Nclima']
        return hashtags
    elif programa =='L6N_20151212':
        hashtags = ['#L6Npizarrapensiones', '#L6Nagrandes', '#L6Nteleblack', '#L6Nvotorogado', '#L6Nanticorrupción']
        return hashtags
    elif programa =='L6N_20151219':
        hashtags = ['#L6Npablopineda', '#L6Nsinluz', '#L6Nbunbury', '#L6Nprecioluz']
        return hashtags
    elif programa =='L6N_20151226':
        hashtags = ['#L6Nblack', '#L6Ncuptalunya', '#L6Npizarraempleo', '#L6Nfranquismo', '#L6Nleche', '#L6Ncampanadas']
        return hashtags
    elif programa =='L6N_20160109':
        hashtags = ['#L6Njuicionóos', '#L6Ncabalgatas', '#L6Nbárcenas']
        return hashtags
    elif programa =='L6N_20160116':
        hashtags = ['#L6Ndesafíocat', '#L6Nverstrynge', '#L6Nretoempleo', '#L6Ncongreso', '#L6Npedroruiz']
        return hashtags
    elif programa =='L6N_20160123':
        hashtags = ['#L6Nestabilidad', '#L6Naute', '#L6Nacuamed', '#L6Nfugacapital']
        return hashtags
    else:
        print('El programa está mal o no tiene hashtags para hacer combinación entre los distintos!!')

def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            dataset.append(row.split("\t"))
    return dataset

def enterFilter(data):
    for row in data:
        row[2] = row[2].replace('\n', '')
    return data

def dataset_tweets_hashtag(data, hashtags):
    dataset = []
    for row in data:
        if row[1] in hashtags:
            dataset.append(row[1:3])
        else:
            continue
    return dataset

def take_pairs(data, hashtags):
    dataset = []
    data = dataset_tweets_hashtag(data, hashtags)
    for row1 in data:
        for row2 in data:
            if row1[0] == row2[0]:
                continue
            else:
                list = [row1[0], row2[0], row1[1], row2[1], 0]
                dataset.append(list)
    return dataset

def datasets_for_pairs(data, hashtags):
    dataset= []
    for row in data:
        if row[0] in hashtags:
            dataset.append(row)
        else:
            continue
    return dataset
'''
def take_pairs(data, hashtags):
    dataset = []
    data = dataset_tweets_hashtag(data, hashtags)
    if len(hashtags) % 2 ==0:
        data1 = datasets_for_pairs(data, hashtags[:len(hashtags)//2])
        data2 = datasets_for_pairs(data, hashtags[len(hashtags)//2:])
    else:
        data1 = datasets_for_pairs(data, hashtags[:round(len(hashtags)//2)])
        data2 = datasets_for_pairs(data, hashtags[math.trunc(len(hashtags)//2):])
    for row1 in data1:
        for row2 in data1:
            if row1[0] == row2[0]:
                continue
            else:
                list = [row1[0], row2[0], row1[1], row2[1], 0]
                dataset.append(list)
    for row1 in data2:
        for row2 in data2:
            if row1[0] == row2[0]:
                continue
            else:
                list = [row1[0], row2[0], row1[1], row2[1], 0]
                dataset.append(list)
    return dataset
'''
def toPandas(data, programa, hashtags):
    file_path = '../bert_data/data_for_embeddings/' + programa + '_' + name_enfoque + '-dist_pairs.txt'
    df = pd.DataFrame(data, columns= ["hashtag1", "hashtag2", "tweet1", "tweet2", "label"])
    df = df[df.hashtag1 != hashtags[-1]]
    if len(data) < 10000:
        #df = df.sample(frac=0.5)
        df = df
    else:
        df = df.sample(n=10000)
    df.to_csv(os.path.relpath(file_path, cur_path), index=False, header=None, sep='\t', doublequote=False)

data = parseDataset(dataset_file)
hashtags_list = get_hashtags(programa)
dataa = dataset_tweets_hashtag(data, hashtags_list)
pairs = take_pairs(data, hashtags_list)

toPandas(pairs, programa, hashtags_list)

#print(len(pairs))
