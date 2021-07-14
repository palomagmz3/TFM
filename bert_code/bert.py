from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import ssl
import os
import re
#import pandas as pd
import itertools
import numpy as np
import sys
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print('TOKENIZERS_PARALLELISM=', os.environ['TOKENIZERS_PARALLELISM'])

programa = 'L6N_20151031'
sys.path.append('../')

path_to_file = '../bert_data/data_for_bert/' + programa + '.txt'
path_to_embeddings = '../bert_data/data_for_embeddings/final/' + 'L6N_ALL-all_pairs.txt'

cur_path = os.path.dirname(__file__)
pairs_train_embeddings_file = os.path.relpath(path_to_embeddings, cur_path)
dataset = os.path.relpath(path_to_file, cur_path)

def parse_embeddings(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            dataset.append(row.split("\t"))
    for row in dataset:
        row[4] = row[4].replace('\n', '')
    return dataset

def list_to_embeddings(data):
    list_with_pairs = []
    for row in data:
        example = InputExample(texts=[row[2], row[3]], label=float(row[4]))
        list_with_pairs.append(example)
    return list_with_pairs

pairs_train_embeddings = parse_embeddings(pairs_train_embeddings_file)
train_examples = list_to_embeddings(pairs_train_embeddings)

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
def modelEmbeddings(data_1half, data_2half, train_examples):
    #sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")

    #Más potente
    sentence_model = SentenceTransformer("dccuchile/bert-base-spanish-wwm-cased")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(sentence_model)
    # Tune the model
    print('Se empieza a entrenar el modelo con nuevos embeddings')
    sentence_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    print('Modelo ya entrenado con nuevos embeddings')
    embeddings_1half = sentence_model.encode(data_1half, show_progress_bar=True)
    print('Primera mitad de los datos procesados...')
    embeddings_2half = sentence_model.encode(data_2half, show_progress_bar=True)
    print('Embeddings hechos')
    print('Definición de modelo vectorial para quitar las stopwords')
    vectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words=stopwords.words('spanish'))
    # Create topic model
    #topic_model = BERTopic(language='spanish', top_n_words=20, nr_topics=topic_reduction, min_topic_size=10, verbose=True)
    topic_model = BERTopic(language='spanish', top_n_words=20, min_topic_size=100, nr_topics=20, low_memory=True, calculate_probabilities=True, vectorizer_model=vectorizer_model, verbose=True)
    print('Se ha cargado el modelo con BERTopic y los parámetros')

    print('Empieza primer fit ttransform')
    topics, probs = topic_model.fit_transform(data_1half + data_2half, np.concatenate((embeddings_1half, embeddings_2half)))
    print('Termina fit transform')
    #print('Se actualizan los topics')
    #new_topics, new_probs = topic_model.reduce_topics(data_1half + data_2half, topics, probs, nr_topics=15)
    print(topic_model.get_topics())
    filename = 'sentence_model_fitted.sav'
    pickle.dump(sentence_model, open(filename, 'wb'))
    print('Se ha guardado el modelo de sentence transformer con los embeddings entrenados')
    topic_model.save("my_model")

    return topics, probs

def matrix(topics, probs):
    matrix = []
    for i in range(len(topics)):
        matrix.append([topics[i]])
        matrix[i].append(probs[i])
    return matrix

def writeFile(matrix):
    mat = np.matrix(matrix)
    file_path = '../bert_data/data_from_bert/' + programa + '_topics_and_probs.txt'
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

data = parseDataset(dataset)
half_data = halfData(data)
other_half = otherhalf(data)
data_for_BERT = dataforBert(data)
print('Se han preparado los datos')
data_1half = dataforBert(half_data)
data_2half = dataforBert(other_half)
topics, probs = modelEmbeddings(data_1half, data_2half, train_examples)
mat = matrix(topics, probs)
writeFile(mat)
#writeTopics(topics)

