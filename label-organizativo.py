import re
import os
import pandas as pd

programa = 'L6N-20151226' #un programa de L6N

ROOT_DIR = os.path.abspath(os.curdir)
PROGRAM_DIR = os.path.join(ROOT_DIR, "programas")
DATA_DIR = os.path.join(ROOT_DIR, "datasets")
dataset_path = programa + '.txt'
dataset_file = os.path.join(PROGRAM_DIR, dataset_path)

from labels_enfoques import labels_organizativo

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

def label (name, data):
    label = ''
    for key in labels_organizativo:
        if (name == key):
            label = labels_organizativo[key]
        else:
            continue
    for row in data:
        row[1] = label
    return data

def toPandas(data):
    return pd.DataFrame(data, columns= ["number", "label", "text"])

def generate_splits(data):
    data["name"] = ["dummy_name" + str(i) for i in range(len(data))]
    dir_name = programa.replace('-', '_')
    path = 'organizativo/all/' + dir_name + '-ALL.txt'
    data.to_csv(os.path.join(DATA_DIR, path), index=False, header=None, sep='\t', doublequote=False)

dataparse = parseDataset(dataset_file)
dateRemoved = datetimeRemoval(dataparse)
dataMerged = mergeUserTweet(dateRemoved)
dataFilteredHashtag = datasetWithoutHashtag(dataMerged)
dataFilteredQuotes = datasetWithoutQuotationMarks(dataFilteredHashtag)
dataFilteredEnter = datasetWithoutEnter(dataFilteredQuotes)
dataLabeled = label(programa, dataFilteredEnter)
df = toPandas(dataLabeled)
generate_splits(df)


