#from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import ssl
import os
import re
#import pandas as pd
import itertools
import numpy as np
import sys
#from sentence_transformers import SentenceTransformer

programa = 'L6N_20151024'
sys.path.append('../')

path_to_file = '../datasets/' + programa + '/distintivo/' + programa + '-ALL.txt'

cur_path = os.path.dirname(__file__)

new_path = os.path.relpath(path_to_file, cur_path)

dataset = []
with open(new_path, 'r') as my_file:
    for row in my_file:
        # añadimos cada fila al dataset pero haciendo un split por donde está la tabulacion
        dataset.append(row.split("\t"))


ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "datasets")
PROGRAM_DIR = os.path.join(ROOT_DIR, "programas")

dataset_path = programa + '/distintivo/' + programa +'-ALL.txt'
dataset_file = os.path.join(DATA_DIR, dataset_path)

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

def halfData(data):
    return data[::2]

def otherhalf(data):
    return data[1::2]

def dataforBert(data):
    l = list(itertools.chain.from_iterable(data))
    new_l = []
    for row in l:
        new_l.append(re.sub('\s+', '-', row))
    return new_l

data = parseDataset(dataset)
only_tweet = onlyTweet(data)
half_data = halfData(only_tweet)
other_half = otherhalf(only_tweet)
data_for_BERT = dataforBert(only_tweet)
data_1half = dataforBert(half_data)
data_2half = dataforBert(other_half)


# Miguel model
def defineModel(n_neighbors, min_topic_size):
    print('CREATED NEW MODEL')
    return BERTopic(language='spanish', n_neighbors=n_neighbors, min_topic_size=min_topic_size, verbose=True)


def basicModel(data):
    topic_model = BERTopic(language='spanish', top_n_words=20, min_topic_size=20, verbose=True)
    topic_model.save("my_model")
    topics, probs = topic_model.fit_transform(data)
    # Update topics
    new_topics, new_probs = topic_model.reduce_topics(data, topics, probs, nr_topics=11)
    print(topic_model.get_topics())


#Create model with custom embeddings
def modelEmbeddings(data_1half, data_2half):
    sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")

    #Más potente
    sentence_model = SentenceTransformer("dccuchile/bert-base-spanish-wwm-uncased")
    embeddings_1half = sentence_model.encode(data_1half, show_progress_bar=True)
    embeddings_2half = sentence_model.encode(data_2half, show_progress_bar=True)
    # Create topic model
    #topic_model = BERTopic(language='spanish', top_n_words=20, nr_topics=topic_reduction, min_topic_size=10, verbose=True)
    topic_model = BERTopic(language='spanish', top_n_words=20, min_topic_size=10,
                           verbose=True)
    topics, probs = topic_model.fit_transform(data_1half + data_2half, np.concatenate((embeddings_1half, embeddings_2half)))
    # Update topics
    new_topics, new_probs = topic_model.reduce_topics(data_1half + data_2half, topics, probs, nr_topics=11)
    print(topic_model.get_topics())
    return new_topics, new_probs

def matrix(topics, probs):
    matrix = []
    for i in range(len(topics)):
        matrix.append([topics[i]])
        matrix[i].append(probs[i])
    return matrix

'''
topics = [22, 6, -1, 22, -1, 6, 38, 56, 38, -1]
probs = [[0, 0, 0.009, 0.147],
       [0, 0, 0.009, 0.147],
       [0, 0, 0.009, 0.147],
[0, 0, 0.009, 0.147],
       [0, 0, 0.009, 0.147],
       [0, 0, 0.009, 0.147],
[0, 0, 0.009, 0.147],
       [0, 0, 0.009, 0.147],
       [0, 0, 0.009, 0.147],
[0, 0, 0.009, 0.147]]
'''

def writeFile(matrix):
    mat = np.matrix(matrix)
    file_path = '../bert_data/data_from_bert/' + programa + '.txt'
    path = os.path.relpath(file_path, cur_path)
    #with open('bert_data/data_from_bert/output.txt', 'wb') as file:
    with open(path, 'wb') as file:
        for line in mat:
            np.savetxt(file, line, fmt='%s')

topics, probs = modelEmbeddings(data_1half, data_2half)
mat = matrix(topics, probs)
print(mat)
writeFile(mat)

