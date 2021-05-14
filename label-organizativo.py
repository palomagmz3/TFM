import re
import os
import pandas as pd

programa = 'L6N-20160123' #un programa de L6N

ROOT_DIR = os.path.abspath(os.curdir)
PROGRAM_DIR = os.path.join(ROOT_DIR, "programas")
DATA_DIR = os.path.join(ROOT_DIR, "datasets")
dataset_path = programa + '.txt'
dataset_file = os.path.join(PROGRAM_DIR, dataset_path)

from labels_enfoques import labels_organizativo
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

def choose_hashtags(program):
    if (program == 'L6N-20151024'):
        return hashtags_L6N20151024
    elif (program == 'L6N-20151114'):
        return hashtags_L6N20151114
    elif (program == 'L6N-20151121'):
        return hashtags_L6N20151121
    elif (program == 'L6N-20151128'):
        return hashtags_L6N20151128
    elif (program == 'L6N-20151205'):
        return hashtags_L6N20151205
    elif (program == 'L6N-20151212'):
        return hashtags_L6N20151212
    elif (program == 'L6N-20151219'):
        return hashtags_L6N20151219
    elif (program == 'L6N-20151226'):
        return hashtags_L6N20151226
    elif (program == 'L6N-20160102'):
        return hashtags_L6N20160102
    elif (program == 'L6N-20160109'):
        return hashtags_L6N20160109
    elif (program == 'L6N-20160116'):
        return hashtags_L6N20160116
    elif (program == 'L6N-20160123'):
        return hashtags_L6N20160123
    else:
        print('El programa seleccionado no existe o no tiene hashtags!!')
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
        else:
            continue
    if (len(labels) == 0):
        return default
    elif (len(labels) == 1):
        return ','.join(labels)
    else:
        winner = labels[0]
        for key in hashtagsprogram:
            if (key in labels and hashtagsprogram[key]>hashtagsprogram[winner]):
                winner = key
            else:
                continue
        return winner

def labelsdata (hashtagsprogram, data):
    for row in data:
        row[1] = label(hashtagsprogram, row[1])
    return data

def toPandas(data):
    return pd.DataFrame(data, columns= ["number", "label", "text"])

def generate_splits(data):
    data["name"] = ["dummy_name" + str(i) for i in range(len(data))]
    dir_name = programa.replace('-', '_')
    path = dir_name + '/aglomerativo/' + dir_name + '-L6N_ALL.txt'
    data.to_csv(os.path.join(DATA_DIR, path), index=False, header=None, sep='\t', doublequote=False)

dataparse = parseDataset(dataset_file)
dateRemoved = datetimeRemoval(dataparse)
dataMerged = mergeUserTweet(dateRemoved)
dataFilteredHashtag = datasetWithoutHashtag(dataMerged)
dataFilteredQuotes = datasetWithoutQuotationMarks(dataFilteredHashtag)
dataFilteredEnter = datasetWithoutEnter(dataFilteredQuotes)

dataLabeled = labelsdata(choose_hashtags(programa), dataFilteredEnter)
df = toPandas(dataLabeled)
generate_splits(df)


