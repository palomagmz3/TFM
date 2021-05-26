import re
import os
import sys
import pandas as pd

programa = 'L6N_ALL'
enfoque = 'aglomerativo' #'distintivo'
sys.path.append('../')

ROOT_DIR = os.path.abspath(os.curdir)
name_enfoque = 'DIST' if enfoque == 'distintivo' else 'AGLO'
path_to_file = '../datasets/' + programa + '/' + enfoque + '/' + programa + '_' + name_enfoque + '_ORIG.txt'
cur_path = os.path.dirname(__file__)

dataset_file = os.path.relpath(path_to_file, cur_path)
dataset = []
def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            #añadimos cada fila al dataset pero haciendo un split por donde está la tabulacion
            dataset.append(row.split("\t"))
    return dataset

def onlyTweet(data):
    dataset = []
    for row in data:
        dataset.append(row[2])
    return dataset

# Método para quitar la parte http de todos los tweets
def removehttp(data):
    dataset = []
    for row in data:
        tweet_split = row.split(' ')
        new_tweet_split = []
        for item in tweet_split:
            if (re.match('(ht+)', item) or item == ''):
                continue
            else:
                new_tweet_split.append(item)
        dataset.append(' '.join(new_tweet_split))
    return dataset

def tweetForBert1(tweet):
    tweet_split = tweet.split(' ')
    if tweet_split[1] == 'RT':
        return ' '.join(tweet_split[3:])
    else:
        return ' '.join(tweet_split[1:])

#Elegir este método si solo queremos quitar el usuario del principio del tweet, aunque nos salgan filas iguales
def dataForBert1(data):
    dataset = []
    for row in data:
        dataset.append(tweetForBert1(row))
    return dataset

#Elegir este método para eliminar las filas repetidas (las que tienen RT)
def dataForBert2(data):
    dataset = []
    for row in data:
        tweet_split = row.split(' ')
        if (len(tweet_split) == 1):
            continue
        elif (tweet_split[1] == 'RT'):
            continue
        else:
            dataset.append(' '.join(tweet_split[1:]))
    return dataset


# Método para quitar los tweets que al final solo conservan un enlace http porque son un RT
# Usar en caso de no haber usado antes el método removehttp
def httpFilter(data):
    dataset = []
    for row in data:
        tweet_split = row.split(' ')
        #print(tweet_split[1])
        new_tweet_split = []
        print(tweet_split)
        for item in tweet_split:
            if item=='':
                print('hay espacio en blanco')
                continue
            else:
                print('no hay espacio en blanco')
                new_tweet_split.append(item)
        if (len(new_tweet_split) == 0):
            continue
        else:
            if (re.match('(https+)',new_tweet_split[0])):
                print('true' + str(tweet_split))
                continue
            else:
                #print('false')
                dataset.append(row)
    return dataset

def toPandas(data):
    file_path = '../bert_data/data_for_bert/' + programa + '_' + name_enfoque + '.txt'
    data_to_pandas = pd.DataFrame(data, columns= ["text"])
    data_to_pandas.to_csv(os.path.relpath(file_path, cur_path), index=False, header=None, sep='\t', doublequote=False)

'''

Aquí empiezan las llamadas a las funciones que hemos definido

'''

data = parseDataset(dataset_file)
data_only_tweet = onlyTweet(data)
#data_without_http = removehttp(data_only_tweet)
#data_for_BERT = dataForBert1(data_without_http)
toPandas(data_only_tweet)
