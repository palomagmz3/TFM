import re
import csv
import pandas as pd
import os
import sys
import numpy as np

programa = 'L6N_20151031'
sys.path.append('../')

cur_path = os.path.dirname(__file__)
path = '../bert_data/data_from_bert/' + programa + '_topics_and_probs.txt'
dataset_file = os.path.relpath(path, cur_path)

def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            #añadimos cada fila al dataset pero haciendo un split por donde está la tabulacion
            row = row.replace('\n', '')
            dataset.append(row)
    return dataset

def flag(string):
    flag = 0
    for item in string:
        if item == '[':
            flag = 1
            return flag
        elif item == ']':
            flag = 2
            return flag
        else:
            continue
    return flag

def replacer(row):
    row = row.replace('0. ', '0 ')
    row = row.replace('1. ', '1 ')
    row = row.replace(' ]', ']')
    row = re.sub(r'[\[\]\(\):,]', r'', row)
    row = re.sub(r'(\s){2,}', r' ', row)
    row_split = row.split(' ')
    return ', '.join(row_split)

def refine_dots(x):
    if (x.endswith('.') or x.endswith(',')):
        x = x[:-1]
    elif (x.endswith(' ')):
        x = ','.join(x.split(',')[:-1])
    return x

def process_with_pandas(dataset):
    df = pd.read_csv(dataset, header=None)
    df['aux'] = df[0].apply(lambda x: flag(x))
    print('Añadida nueva columna para indicar principio de probabilidades de un tweet')
    df['join'] = np.nan
    print('Añadida nueva columna vacía para la configuración de grupos')
    df.at[0, 'join'] = 3
    start = 3
    print('Se empiezan a formar los grupos')
    for index in range(df.shape[0]):
        if df.iloc[index]['aux'] == 1:
            start += 1
            df.at[index, 'join'] = start
        elif df.iloc[index]['aux'] != 1:
            df.at[index, 'join'] = start
    print('Grupos creados')
    df = df[[0, 'join']]
    print('Se crean los grupos')
    df = df.groupby('join', sort=False)[0].apply(' '.join).reset_index(name='new_seq')
    print('Se realiza procesado de texto para remplazar los caracteres inválidos')
    df['new_seq'] = df['new_seq'].apply(lambda x: replacer(x))
    df['new_seq'] = df['new_seq'].apply(lambda x: refine_dots(x))
    df.drop('join', inplace=True, axis=1)
    file_path = '../bert_data/data_from_bert_processed/' + programa + '.csv'
    df.to_csv(os.path.relpath(file_path, cur_path), index=False, header=None, sep='\t', doublequote=False, quoting=csv.QUOTE_NONE)
    print('Fin. Ya está el csv listo')

def processDatasetWords(data):
    dataset = []
    for row in data:
        row = re.sub(r'[\[\]\(\):,]', r'', row)
        row_split = row.split(' ')[0::2]
        dataset.append(', '.join(row_split))
    return dataset

def processMatrix(data):
    dataset = []
    for row in data:
        row = re.sub(r'[\[\]\(\):,]', r'', row)
        row_split = row.split(' ')
        dataset.append(', '.join(row_split))
    return dataset

def toCSV(data):
    file_path = '../bert_data/data_from_bert_processed/' + programa + '.csv'
    df = pd.DataFrame(data, columns=['text'])
    df.to_csv(os.path.relpath(file_path, cur_path), index=False, header=None, sep='\t', doublequote=False, quoting=csv.QUOTE_NONE)

#data = parseDataset(dataset_file)
#dataa = process_with_pandas(data)
#print(dataa)
#data_for_visualize = processMatrix(data)
#toCSV(data_for_visualize)
process_with_pandas(dataset_file)
