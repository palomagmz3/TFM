import re
import pandas as pd
import os
import sys

sys.path.append('../')
cur_path = os.path.dirname(__file__)

programa = 'L6N_ALL'
enfoque = 'distintivo'

name_enfoque = 'DIST' if enfoque == 'distintivo' else 'AGLO'
path_to_file = '../bert_data/data_for_bert/' + programa + '_' + name_enfoque +  '-orig_labels.txt'

dataset_file = os.path.relpath(path_to_file, cur_path)

def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            dataset.append(row.split("\t"))
    return dataset

def enterFilter(data):
    for row in data:
        row[1] = row[1].replace('\n', '')
    return data

def hashtags(data):
    hashtags = []
    for row in data:
        if row[0] in hashtags:
            continue
        else:
            hashtags.append(row[0])
    return hashtags

def dataset_tweets_hashtag(data, item):
    dataset = []
    for row in data:
        if (row[0] == item):
            dataset.append(row)
        else:
            continue
    return dataset

def take_pairs(data, hashtags):
    dataset  = []
    for item in hashtags:
        tweets_same_hashtags = dataset_tweets_hashtag(data, item)
        if len(tweets_same_hashtags) % 2 == 0:
            continue
        else:
            tweets_same_hashtags.pop()
        tweets_same_hashtags1 = tweets_same_hashtags[::2]
        tweets_same_hashtags2 = tweets_same_hashtags[1::2]
        for i in range(len(tweets_same_hashtags1)):
            list = [tweets_same_hashtags1[i][0], tweets_same_hashtags2[i][0], tweets_same_hashtags1[i][1], tweets_same_hashtags2[i][1], 1]
            dataset.append(list)
    return dataset

def toPandas(data, programa):
    file_path = '../bert_data/data_for_bert/' + programa + '_' + name_enfoque + '-eq_pairs.txt'
    data_to_pandas = pd.DataFrame(data)
    data_to_pandas.to_csv(os.path.relpath(file_path, cur_path), index=False, header=None, sep='\t', doublequote=False)

data = parseDataset(dataset_file)
print(len(data))
data = enterFilter(data)
hashtags_list = hashtags(data)
pairs = take_pairs(data, hashtags_list)
toPandas(pairs, programa=programa)
