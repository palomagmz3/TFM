from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import ssl
import re
#import pandas as pd
import itertools
import numpy as np
from sentence_transformers import SentenceTransformer

dataset_file = "L6N-20151128.txt"

def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            #añadimos cada fila al dataset pero haciendo un split por donde está la tabulacion
            dataset.append(row.split("\t"))
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

data = parseDataset(dataset_file)
half_data = halfData(data)
other_half = otherhalf(data)
data_for_BERT = dataforBert(data)
data_1half = dataforBert(half_data)
data_2half = dataforBert(other_half)

'''
# Miguel model
def defineModel(n_neighbors, min_topic_size):
    print('CREATED NEW MODEL')
    return BERTopic(language='spanish', n_neighbors=n_neighbors, min_topic_size=min_topic_size, verbose=True)

'''

def basicModel(data):
    topic_model = BERTopic(language='spanish', top_n_words=20, min_topic_size=20, verbose=True)
    topic_model.save("my_model")
    topics, probs = topic_model.fit_transform(data)
    # Update topics
    new_topics, new_probs = topic_model.reduce_topics(data, topics, probs, nr_topics=11)
    print(topic_model.get_topics())


#Create model with custom embeddings
def modelEmbeddings(data):
    sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")

    #Más potente
    sentence_model = SentenceTransformer("dccuchile/bert-base-spanish-wwm-uncased")
    embeddings_1half = sentence_model.encode(data_1half, show_progress_bar=True)
    embeddings_2half = sentence_model.encode(data_2half, show_progress_bar=True)
    # Create topic model
    topic_model = BERTopic(language='spanish', top_n_words=20, min_topic_size=20, verbose=True)
    topics, probs = topic_model.fit_transform(data_1half + data_2half, np.concatenate((embeddings_1half, embeddings_2half)))
    # Update topics
    new_topics, new_probs = topic_model.reduce_topics(data_1half + data_2half, topics, probs, nr_topics=11)
    print(topic_model.get_topics())

def matrix(topics, probs):
    matrix = []
    for i in range(len(topics)):
        matrix.append([topics[i]])
        matrix[i].append(probs[i])
    return matrix

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

mat = matrix(topics, probs)
print(mat)

def writeFile(matrix):
    mat = np.matrix(matrix)
    with open('bert_data/data_from_bert/output.txt', 'wb') as file:
        for line in mat:
            np.savetxt(file, line, fmt='%s')

writeFile(mat)

