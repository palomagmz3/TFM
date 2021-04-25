import re
import pandas as pd
import os
import sys

programa = 'L6N_20151024'
sys.path.append('../')

ROOT_DIR = os.path.abspath(os.curdir)
path_to_file = '../datasets/' + programa + '/distintivo/' + programa + '-ALL.txt'
cur_path = os.path.dirname(__file__)

dataset_file = os.path.relpath(path_to_file, cur_path)


def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            dataset.append(row.split("\t"))
    return dataset

def onlyTweetAndLabel(data):
    dataset = []
    for row in data:
        dataset.append(row[1])
    return dataset

def removehttpttweet(tweet):
    tweet_split = tweet.split(' ')
    new_tweet_split = []
    for item in tweet_split:
        if (re.match('(ht+)', item) or item == ''):
            continue
        else:
            new_tweet_split.append(item)
    return ' '.join(new_tweet_split)

# Método para quitar la parte http de todos los tweets
def removehttp(data):
    for row in data:
        row[1] = removehttpttweet(row[1])
    return data

#Elegir este método para eliminar las filas repetidas (las que tienen RT)
def dataForBert2(data):
    dataset = []
    for row in data:
        tweet_split = row[1].split(' ')
        if (len(tweet_split) == 1):
            continue
        elif (tweet_split[1] == 'RT'):
            continue
        else:
            row[1]=' '.join(tweet_split[1:])
            dataset.append(row)
    return dataset

def toPandas(data):
    file_path = '../bert_data/data_with_label/' + programa + '.txt'
    data_to_pandas = pd.DataFrame(data, columns= ["label", "text"])
    data_to_pandas.to_csv(os.path.relpath(file_path, cur_path), index=False, header=None, sep='\t', doublequote=False)

dataset = parseDataset(dataset_file)
data_with_label = onlyTweetAndLabel(dataset) #para quedarnos con el tweet y el hashtag
#data_without_http = removehttp(data_with_label)
#data_bert_with_hashtag = dataForBert2(data_without_http)
#toPandas(data_with_label)
