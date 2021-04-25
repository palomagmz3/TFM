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

path_to_file = '../bert_data/data_for_bert' + programa + '.txt'

cur_path = os.path.dirname(__file__)

dataset = os.path.relpath(path_to_file, cur_path)

def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
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

data = parseDataset(dataset)
half_data = halfData(data)
other_half = otherhalf(data)
data_for_BERT = dataforBert(data)
data_1half = dataforBert(half_data)
data_2half = dataforBert(other_half)

print('Se han preparado los datos')

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
    print('Embeddings hechos')
    # Create topic model
    #topic_model = BERTopic(language='spanish', top_n_words=20, nr_topics=topic_reduction, min_topic_size=10, verbose=True)
    topic_model = BERTopic(language='spanish', top_n_words=20, min_topic_size=10, low_memory=True, calculate_probabilities=False, verbose=True)
    print('Se ha cargado el modelo con BERTopic y los parámetros')
    topic_model.save("my_model")
    print('Empieza primer fit ttransform')
    topics, probs = topic_model.fit_transform(data_1half + data_2half, np.concatenate((embeddings_1half, embeddings_2half)))
    # Update topics
    print('Se actualizan los topics')
    new_topics, new_probs = topic_model.reduce_topics(data_1half + data_2half, topics, probs, nr_topics=11)
    print(topic_model.get_topics())
    return new_topics, new_probs

def matrix(topics, probs):
    matrix = []
    for i in range(len(topics)):
        matrix.append([topics[i]])
        matrix[i].append(probs[i])
    return matrix

def writeFile(matrix):
    mat = np.matrix(matrix)
    file_path = '../bert_data/data_from_bert/' + programa + '.txt'
    path = os.path.relpath(file_path, cur_path)
    #with open('bert_data/data_from_bert/output.txt', 'wb') as file:
    with open(path, 'wb') as file:
        for line in mat:
            np.savetxt(file, line, fmt='%s')

def writeTopics(topics):
    file_path = '../bert_data/data_from_bert/' + programa + '.txt'
    path = os.path.relpath(file_path, cur_path)
    with open(path, 'w') as file:
        for item in topics:
            file.write("%s\n" % item)

topics, probs = modelEmbeddings(data_1half, data_2half)
#mat = matrix(topics, probs)
#writeFile(mat)
writeTopics(topics)
