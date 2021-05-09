import re
import pandas as pd
import os
import sys

#programa = 'L6N_20151024'
sys.path.append('../')
cur_path = os.path.dirname(__file__)

'''
ROOT_DIR = os.path.abspath(os.curdir)
path_to_file = '../datasets/' + programa + '/distintivo/' + programa + '-ALL.txt'
cur_path = os.path.dirname(__file__)

dataset_file = os.path.relpath(path_to_file, cur_path)
'''
def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            dataset.append(row.split("\t"))
    return dataset

def onlyTweetAndLabel(data):
    dataset = []
    for row in data:
        dataset.append(row[1])
    return dataset

def to_visualize(bert_data, labeled_data):
    # Salida es lo mismo que sale de Bert pero la primera columna ya no es un número de topic sino un hashtag
    # Usamos el método tanto para la etiqueta nueva de Bert como para la original
    if (len(bert_data) != len(labeled_data)):
        print('El tamaño de los tweets y la salida de bert no coincide!!')
    else:
        for i in range(len(labeled_data)):
            list = []
            list.append(labeled_data[i])
            new_row = labeled_data[i] + ',' + ','.join(bert_data[i].split(',')[1:])
            bert_data[i] = new_row
    return bert_data

def toPandas(data, programa):
    file_path = '../bert_data/data_to_visualize/' + programa + '-orig_labels.csv'
    data_to_pandas = pd.DataFrame(data)
    data_to_pandas.to_csv(os.path.relpath(file_path, cur_path), index=False, header=None, sep='\t', doublequote=False)

'''
dataset = parseDataset(dataset_file)
data_with_label = onlyTweetAndLabel(dataset) #para quedarnos con el tweet y el hashtag
#data_without_http = removehttp(data_with_label)
#data_bert_with_hashtag = dataForBert2(data_without_http)
#toPandas(data_with_label)
'''