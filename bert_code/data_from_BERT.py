import re
import csv
import pandas as pd
import os
import sys
import numpy as np

programa = 'L6N_ALL_AGLO'
num_topics = 100
embeddings = True

e = '_embeddings' if embeddings==True else ''
n = '' if num_topics =='' else '_' + str(num_topics)
sys.path.append('../')

cur_path = os.path.dirname(__file__)
path = '../bert_data/data_from_bert/' + programa + '_topics_and_probs' + e + n + '.txt'
print(path)
dataset_file = os.path.relpath(path, cur_path)

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
    file_path = '../bert_data/data_from_bert_processed/' + programa + e + n +'.csv'
    df.to_csv(os.path.relpath(file_path, cur_path), index=False, header=None, sep='\t', doublequote=False, quoting=csv.QUOTE_NONE)
    print('Fin. Ya está el csv listo')

process_with_pandas(dataset_file)
