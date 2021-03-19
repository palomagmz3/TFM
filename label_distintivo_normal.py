import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PATH, "datasets")

dataset_file = "programas/L6N-20151128.txt"

from labels_enfoques import hashtags_L6N20151024, \
    hashtags_L6N20151114, \
    hashtags_L6N20151121, \
    hashtags_L6N20151128, \
    hashtags_L6N20151205, \
    hashtags_L6N20151212, \
    hashtags_L6N20151219, \
    hashtags_L6N20151226, \
    hashtags_L6N20160102, \
    hashtags_L6N20160109, \
    hashtags_L6N20160116, \
    hashtags_L6N20160123

def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            #añadimos cada fila al dataset pero haciendo un split por donde está la tabulacion
            dataset.append(row.split("\t"))
    return dataset

def datetimeRemoval(data):
    for row in data:
        del row[1]
    return data

def mergeUserTweet(data):
    for row in data:
        row[2:4] = [' '.join(row[2:4])]
    return data

def hashtagFilter(sentence):
    return re.sub(r'(#[A-Za-z0-9\_áéíóúüñÁÉÍÓÚÜÑ]+)', r'', sentence)

def datasetWithoutHashtag(data):
    for row in data:
        row[2] = hashtagFilter(row[2])
    return data

def quotationMarksFilter(sentence):
    return sentence.replace("'", '').replace('"', '')

def datasetWithoutQuotationMarks(data):
    for row in data:
        row[2] = quotationMarksFilter(row[2])
    return data

def enterFilter(sentence):
    return sentence.replace('\n', '')

def datasetWithoutEnter(data):
    for row in data:
        row[2] = enterFilter(row[2])
    return data

def label (hashtagsprogram, labels):
    labels = labels.split(',')
    default = max(hashtagsprogram.items(), key=lambda x:x[1])[0]
    if (len(labels) == 0):
        return default
    for label in labels:
        if label not in hashtagsprogram:
            labels.remove(label)
    if (len(labels) == 0):
        return default
    if (len(labels) == 1):
        return ','.join(labels)
    else:
        winner = labels[0]
        for key in hashtagsprogram:
            if (key in labels and hashtagsprogram[key]<hashtagsprogram[winner]):
                winner = key
        return winner

def labelsdata (hashtagsprogram, data):
    for row in data:
        row[1] = label(hashtagsprogram, row[1])
    return data

def toPandas(data):
    return pd.DataFrame(data, columns= ["number", "label", "text"])


def generateSplits(data):
    data["name"] = ["dummy_name" + str(i) for i in range(len(data))]
    train, test = train_test_split(data, test_size=0.3)
    data.to_csv(os.path.join(
        DATA_DIR, 'L6N-20151128/distintivo', 'all.txt'), index=False, header=None, sep='\t', doublequote=False)
    train.to_csv(os.path.join(
        DATA_DIR, 'L6N-20151128/distintivo', 'training.txt'), index=False, header=None, sep='\t', doublequote=False)
    test.to_csv(os.path.join(
        DATA_DIR, 'L6N-20151128/distintivo', 'test.txt'), index=False, header=None, sep='\t')

'''

Aquí empiezan las llamadas a las funciones que hemos definido

'''

# Llamadas para procesar el dataset
dataparse = parseDataset(dataset_file)
dateRemoved = datetimeRemoval(dataparse)
dataMerged = mergeUserTweet(dateRemoved)
dataNoHashtag = datasetWithoutHashtag(dataMerged)
dataNoQuote = datasetWithoutQuotationMarks(dataNoHashtag)
dataNoEnter = datasetWithoutEnter(dataNoQuote)
dataLabeled = labelsdata(hashtags_L6N20151128, dataNoEnter) #Cambiar el primer parámetro

# Llamadas para generear los datasets
df = toPandas(dataLabeled)
#print(df)
generateSplits(df)
