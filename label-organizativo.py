import re

dataset_file = "programas/L6N-20151031.txt"

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

def enterFilter(sentence):
    return sentence.replace('\n', '')

def datasetWithoutEnter(data):
    for row in data:
        row[2] = enterFilter(row[2])
    return data

def label (name, data):
    file = name.split('/')
    label = ''
    for key in labels_organizativo:
        if (file[1] == key):
            label = labels_organizativo[key]
    for row in data:
        row[1] = label
    return data

dataparse = parseDataset(dataset_file)
dateRemoved = datetimeRemoval(dataparse)
dataMerged = mergeUserTweet(dateRemoved)
dataFilteredHashtag = datasetWithoutHashtag(dataMerged)
dataFilteredEnter = datasetWithoutEnter(dataFilteredHashtag)
dataLabeled = label(dataset_file, dataFilteredEnter)

for row in dataLabeled:
    print(row)

