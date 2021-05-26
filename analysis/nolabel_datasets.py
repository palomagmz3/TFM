
'''

Script de python para analizar y etiquetar los dos programas de La Sexta Noche que no están etiquetados

'''

import re
import os
import sys

programa = 'L6N-20151107'
sys.path.append('../')

cur_path = os.path.dirname(__file__)
path = '../programas/' + programa + '.txt'
dataset_file = os.path.relpath(path, cur_path)


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

def gapsRemoval(sentence):
    for item in sentence:
        if (item == ''):
            sentence.remove(item)
    return sentence

def addLabel(data):
    for row in data:
        row_one_space = re.sub('\s+',' ',row[3])
        row_split = row_one_space.split(' ')
        new_row = gapsRemoval(row_split)
        label = []
        for item in new_row:
            if (item[0]=='#'):
                label.append(item)
        if (len(label)==0):
            row[1] = '#NoHashtag'
            #row.insert(1, '#NoHashtag')
        else:
            row[1] = ','.join(label)
            #row.insert(1, ','.join(label))
    return data

def mergeUserTweet(data):
    for row in data:
        row[2:4] = [' '.join(row[2:4])]
    return data


# Aquí empiezan las llamadas para procesar el dataset

parse = parseDataset(dataset_file)
dateRemoved = datetimeRemoval(parse)
dataLabeled = addLabel(dateRemoved)
dataMerged = mergeUserTweet(dataLabeled)

#for row in dataMerged:
#    print(row)

'''
Análisis de las etiquetas de estos programas
'''

#Método para ver qué etiquetas de estos dos programas se repiten más (ordenadas de menos a mayor)
def labels(data):
    dictionary = {}
    dataset = []
    for row in data:
        dataset.append(row[1])
    for row in dataset:
        if (row in dictionary):
            dictionary[row] += 1
        else:
            dictionary[row] = 1
    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}

print(labels(dataMerged))

def multilabelFilter(label):
    nAlmohadilla = 0
    for i in label:
        if (i=='#'):
            nAlmohadilla += 1
    if (nAlmohadilla>1):
        return True
    else:
        return False

def parsehashtags(sentence):
    sentencesplit = sentence.split(',')
    if (len(sentencesplit)>1):
        #print('más de un hashtag')
        for i in range(0, len(sentencesplit)):
            if not 'L6N' in sentencesplit[i]:
                sentencesplit[i] = ''
        return ''.join(sentencesplit)
    else:
        #print('un solo hashtag')
        if not 'L6N' in sentence:
            #re.sub(r'(#[A-Za-z0-9\_áéíóúüñÁÉÍÓÚÜÑ]+)', r'', sentence)
            sentencesplit[0] = ''
            return ''.join(sentencesplit)
        else:
            return ''.join(sentencesplit)


def parsehashtagdata(data):
    for row in data:
       row[1] = parsehashtags(row[1])
    return data

newdata = parsehashtagdata(dataLabeled)


def refinehashtagsL6N20151031(label):
    nAlmohadilla = 0
    for i in label:
        if (i == '#'):
            nAlmohadilla += 1
    if (nAlmohadilla > 1):
        return label
    else:
        if (re.match('(#L6N[Rr][Ee]+)',label)):
            label = 'L6Nretocatalán'
            return label
        elif (re.match('(#L6N[EeCc][Ll]+)',label)):
            label = 'L6Nelclanpujol'
            return label
        elif (re.match('(#L6N[Jj]+)',label)):
            label = 'L6Njuecesgurtel'
            return label
        elif (re.match('(#L6N[AaCc][AaGg]+)',label)):
            label = 'L6Ncallegarzón'
            return label
        elif (re.match('(#L6N[Rr][Ii]+)',label)):
            label = 'L6Nrivera24h'
            return label
        elif (re.match('(#L6N[Ee][Nn]+)',label)):
            label = 'L6Nencampaña'
            return label
        elif (re.match('(#L6N[Bb][Aa]+)',label)):
            label = 'L6Nbalance'
            return label
        else:
            return label


def refinehashtagsdataL6N20151031(data):
    for row in data:
        row[1] = refinehashtagsL6N20151031(row[1])
    return data



def refinehashtagsL6N20151107(label):
    nAlmohadilla = 0
    for i in label:
        if (i == '#'):
            nAlmohadilla += 1
    if (nAlmohadilla > 1):
        return label
    else:
        if (re.match('(#L6N[Rr][Ee]+)',label)):
            label = 'L6Nretocatalán'
            return label
        elif (re.match('(#L6N[Cc][Aa]+)',label)):
            label = 'L6Ncallerivera'
            return label
        elif (re.match('(#L6N[Oo]+)',label)):
            label = 'L6Nobjetivo20D'
            return label
        elif (re.match('(#L6N[Pp][Ii]+)',label)):
            label = 'L6Npizarrasparo'
            return label
        elif (re.match('(#L6N[Ii][Gg]+)',label)):
            label = 'L6Niglesias24h'
            return label
        else:
            return label


def refinehashtagsdataL6N20151107(data):
    for row in data:
        row[1] = refinehashtagsL6N20151107(row[1])
    return data


newdata = refinehashtagsdataL6N20151107(newdata)
#for row in newdata:
#    print(row)

def listlabels(data):
    labels = []
    for row in data:
        if (multilabelFilter(row[1])):
            labelsrow = row[1].split(',')
            for item in labelsrow:
                if item in labels:
                    continue
                else:
                    labels.append(item)
        else:
            if (row[1] in labels):
                continue
            else:
                labels.append(row[1])
    return labels

labels = listlabels(dataLabeled)
print(len(labels))
print(listlabels(newdata))

def map(data):
    labels = {}
    for row in data:
        if row[1] in labels:
            labels[row[1]] += 1
        else:
            labels[row[1]] = 1
    return labels


dic = map(dataLabeled)
print(dic)
print(len(dic))

'''
Etiquetar estos programas con el hashtag elegido:
Programa L6N20151031 - L6Ncallegarzón
Programa L6N20151107 - L6Ncallerivera
'''

def labelL6n20151031(data):
    for row in data:
        row[1] = 'L6Ncallegarzón'
    return data

def labelL6n20151107(data):
    for row in data:
        row[1] = 'L6Ncallerivera'
    return data

#newdata = labelL6n20151031(dataMerged)
newdata = labelL6n20151107(dataMerged)
