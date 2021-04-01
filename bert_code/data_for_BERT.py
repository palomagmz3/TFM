import re
import os
import pandas as pd

ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(ROOT_DIR, "datasets")

BERT_DIR = os.path.join(ROOT_DIR, "bert_data/data_with_label")
dataset_file = os.path.join(DATA_DIR, "L6N_20151128/distintivo/all.txt")

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
    data_to_pandas = pd.DataFrame(data, columns= ["text"])
    data_to_pandas.to_csv(os.path.join(BERT_DIR, 'L6N-20151128.txt'), index=False, header=None, sep='\t', doublequote=False)

'''

Aquí empiezan las llamadas a las funciones que hemos definido

'''

data = parseDataset(dataset_file)
data_only_tweet = onlyTweet(data)
data_without_http = removehttp(data_only_tweet)
data_for_BERT = dataForBert2(data_without_http)
#toPandas(data_for_BERT)

'''

Parte del script para hacer lo mismo que antes pero aplicado a un dataset en el que también queremos los hashtags

'''

def onlyTweetAndLabel(data):
    dataset = []
    for row in data:
        dataset.append(row[1:3])
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
    data_to_pandas = pd.DataFrame(data, columns= ["label", "text"])
    data_to_pandas.to_csv(os.path.join(BERT_DIR, 'L6N-20151128.txt'), index=False, header=None, sep='\t', doublequote=False)

data_with_label = onlyTweetAndLabel(data) #para quedarnos con el tweet y el hashtag
data_without_http = removehttp(data_with_label)
data_bert_with_hashtag = dataForBert2(data_without_http)
toPandas(data_bert_with_hashtag)
for row in data_bert_with_hashtag:
    print(row)

print(len(data_bert_with_hashtag))