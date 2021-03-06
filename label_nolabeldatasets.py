import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from labels_enfoques import hashtags_L6N20151031, hashtags_L6N20151107

programa = 'L6N-20151107'
enfoque = '/distintivo/' #aglomerativo

ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(ROOT_DIR, "datasets")
PROGRAM_DIR = os.path.join(ROOT_DIR, "programas")

dataset_path = programa + '.txt'
dataset_file = os.path.join(PROGRAM_DIR, dataset_path)

def choose_hashtags(program):
    if (program == 'L6N-20151031'):
        return hashtags_L6N20151031
    elif (program == 'L6N-20151107'):
        return hashtags_L6N20151107
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

def parsehashtags(sentence):
    sentencesplit = sentence.split(',')
    if (len(sentencesplit)>1):
        for i in sentencesplit:
            if not 'L6N' in sentencesplit:
                sentencesplit.remove(i)
        return ','.join(sentencesplit)
    else:
        if not 'L6N' in sentence:
            sentencesplit[0] = ''
            return ','.join(sentencesplit)
        else:
            return ','.join(sentencesplit)

def parsehashtagdata(data):
    for row in data:
       row[1] = parsehashtags(row[1])
    return data

def refinehashtagsL6N20151031(labels):
    labelsplit = labels.split(',')
    lista_aux = []
    if len(labelsplit)==0:
        return labelsplit
    #for l in range(len(labelsplit)):
    for l in labelsplit:
        if (re.match('(#L6N[Rr][Ee]+)',l)):
            new = '#L6Nretocatalán'
            lista_aux.append(new)
        elif (re.match('(#L6N[EeCc][Ll]+)',l)):
            new = '#L6Nelclanpujol'
            lista_aux.append(new)
        elif (re.match('(#L6N[Jj]+)',l)):
            new = '#L6Njuecesgurtel'
            lista_aux.append(new)
        elif (re.match('(#L6N[AaCc][AaGg]+)',l)):
            new = '#L6Ncallegarzón'
            lista_aux.append(new)
        elif (re.match('(#L6N[Rr][Ii]+)',l)):
            new = '#L6Nrivera24h'
            lista_aux.append(new)
        elif (re.match('(#L6N[Ee][Nn]+)',l)):
            new = '#L6Nencampaña'
            lista_aux.append(new)
        elif (re.match('(#L6N[Bb][Aa]+)',l)):
            new = '#L6Nbalance'
            lista_aux.append(new)
        else:
            labelsplit.remove(l)
    return ','.join(lista_aux)

x = refinehashtagsL6N20151031('#OTRASheffield3,#TJHalloween,#LaHoraMagica642,#RugbyWorldCup,#L6Nretocatalán,#news')
print(x)

def refinehashtagsdataL6N20151031(data):
    for row in data:
        row[1] = refinehashtagsL6N20151031(row[1])
    return data

def refinehashtagsL6N20151107(label):
    labelsplit = label.split(',')
    lista_aux = []
    if (len(labelsplit)==0):
        return labelsplit
    for l in labelsplit:
        if (re.match('(#L6N[Rr][Ee]+)',l)):
            new = '#L6Nretocatalán'
            lista_aux.append(new)
        elif (re.match('(#L6N[Cc][Aa]+)',l)):
            new = '#L6Ncallerivera'
            lista_aux.append(new)
        elif (re.match('(#L6N[Oo]+)',l)):
            new = '#L6Nobjetivo20D'
            lista_aux.append(new)
        elif (re.match('(#L6N[Pp][Ii]+)',l)):
            new = '#L6Npizarrasparo'
            lista_aux.append(new)
        elif (re.match('(#L6N[Ii][Gg]+)',l)):
            new = '#L6Niglesias24h'
            lista_aux.append(new)
        else:
            labelsplit.remove(l)
    return ','.join(lista_aux)


def refinehashtagsdataL6N20151107(data):
    for row in data:
        row[1] = refinehashtagsL6N20151107(row[1])
    return data

def choose_refine(data, program):
    if (program == 'L6N-20151031'):
        return refinehashtagsdataL6N20151031(data)
    elif (program == 'L6N-20151107'):
        return refinehashtagsdataL6N20151107(data)
    else:
        print('El programa especificado no tiene función para refinar etiquetas!')

def finalLabel (hashtagsprogram, labels):
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
    if (len(labels) == 1):
        return ','.join(labels)
    else:
        winner = labels[0]
        if (enfoque == '/distintivo/'):
            for key in hashtagsprogram:
                if (key in labels and hashtagsprogram[key]<hashtagsprogram[winner]):
                    winner = key
                else:
                    continue
        elif (enfoque == '/aglomerativo/'):
            for key in hashtagsprogram:
                if (key in labels and hashtagsprogram[key]>hashtagsprogram[winner]):
                    winner = key
                else:
                    continue
        else:
            print('Se ha especificado un enfoque inválido!!')
        return winner

def labelsdata (hashtagsprogram, data):
    for row in data:
        row[1] = finalLabel(hashtagsprogram, row[1])
    return data

def toPandas(data):
    return pd.DataFrame(data, columns= ["number", "label", "text"])


def generateSplits(data):
    data["name"] = ["dummy_name" + str(i) for i in range(len(data))]
    dir_name = programa.replace('-', '_')
    path = dir_name + enfoque + dir_name + '-L6N_ALL.txt'
    data.to_csv(os.path.join(DATA_DIR, path), index=False, header=None, sep='\t', doublequote=False)
'''

Aquí empiezan las llamadas a las funciones que hemos definido

'''
# Llamadas para procesar el dataset

parse = parseDataset(dataset_file)
dateRemoved = datetimeRemoval(parse)
dataLabeled = addLabel(dateRemoved)
dataMerged = mergeUserTweet(dataLabeled)
dataNoHashtag = datasetWithoutHashtag(dataMerged)
dataNoQuote = datasetWithoutQuotationMarks(dataNoHashtag)
dataNoEnter = datasetWithoutEnter(dataNoQuote)
dataParseHashtag = parsehashtagdata(dataNoEnter)
#dataRefined = refinehashtagsdataL6N20151107(dataParseHashtag)
dataRefined =choose_refine(dataParseHashtag, programa)
dataFinalLabeled = labelsdata(choose_hashtags(programa), dataRefined)

# Llamadas para generear los datasets
df = toPandas(dataFinalLabeled)
generateSplits(df)

