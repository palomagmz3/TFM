
import re
dataset_file= '../programas/L6N-20160123.txt'
dataset_file2='../programas/L6N-20160102.txt'
dataset_file3='../programas/L6N-20151121.txt'
'''
Este archivo de python es para analizar las filas de los documentos de la L6N que tienen varios hashtags asociados
'''
def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            #añadimos cada fila al dataset pero haciendo un split por donde está la tabulacion
            dataset.append(row.split("\t"))
    return dataset

parse = parseDataset(dataset_file)

print(len(parse))

def multilabel (sentence):
    nhashtag = 0
    for item in sentence[2]:
        if (item=='#'):
            nhashtag += 1
    if (nhashtag>1):
        return True
    else:
        return False

def multilabelDataset (file):
    dataset = []
    for row in file:
        if (multilabel(row)):
            dataset.append(row)
    return dataset

datafiltered = multilabelDataset(parse)

'''
A partir de aquí se realiza un análisis de los hashtags para buscar patrones
1) Hacemos un dataset de hashtags (método onlyHashtag)
2) Lo parseamos para que cada fila sea un array formado por los hashtags que contiene
'''
#Método para quedarnos solo con el campo del hashtag de cada fila para proceder al análisis
def onlyHashtag (file):
    dataset = []
    for row in file:
        dataset.append(row[2])
    return dataset

dataOnlyHashtag = onlyHashtag(datafiltered)

def parseHashtag (file):
    dataset = []
    for row in file:
        dataset.append(row.split(','))
    return dataset

dataOnlyHashtagParsed = parseHashtag(dataOnlyHashtag)

def refine(row, dict):
    row_split = row.split(',')
    keys_list = []
    for key in dict.keys():
        keys_list.append(key.split(','))
    if (len(row_split) == 2):
        for item in keys_list:
            if (len(item) > 2):
                keys_list.remove(item)
        if (len(keys_list) == 0):
            return False, None
        for item in keys_list:
            if (row_split[0] == item[1] and row_split[1] == item[0]):
                return [True, ','.join(item)]
    if (len(row_split) == 3):
        for item in keys_list:
            if (len(item) != 3):
                keys_list.remove(item)
        if (len(keys_list) == 0):
            print('len = 0')
            return False, None
        for item in keys_list:
            print('forrrr')
            if ((row_split[0] == item[0] or row_split[0] == item[1] or row_split[0] == item[2]) and (row_split[1] == item[0] or row_split[1] == item[1] or row_split[1] == item[2]) and (row_split[2] == item[0] or row_split[2] == item[1] or row_split[2] == item[2])):
                return [True, ','.join(item)]
    else:
        return False, None

def map (file):
    dictionary = {}
    for row in file:
        if (row in dictionary):
            #incluir algo que compruebe que está en el diccionario la etiqueta pero con los hashtags desordenados
            dictionary[row] += 1
        #elif (refine(row, dictionary)[0]):
        #    b, item = refine(row, dictionary)
        #    dictionary[item] += 1
        #Otra condición para sumar también si las líneas tienen los mismos hashtags pero en distinto order
        else:
            dictionary[row] = 1
    return dictionary

dictionary = map(dataOnlyHashtag)

#print(dictionary)

#Método para contar el númerro de total de tweets de cada hashtag (UNA ÚNICA ETIQUETA)
def hashtagsindividuales(data):
    dictionary = {}
    dataset = []
    for row in data:
        if (multilabel(row)):
            continue
        else:
            dataset.append(row[2])
    for row in dataset:
        if(row in dictionary):
            dictionary[row] += 1
        else:
            dictionary[row] = 1
    return dictionary

dictOneHashtag = hashtagsindividuales(parse)
print(dictOneHashtag)