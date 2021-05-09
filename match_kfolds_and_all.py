import os
import pandas as pd
import numpy as np
import csv
programa = 'L6N_20160123' #un programa de L6N
enfoque = '/aglomerativo/' #distintivo
n = '1' #un número del 1 al 5 (listas kfold)
t = 'TEST' #TRAIN o TEST

def name_file(enfoque):
    if enfoque == '/aglomerativo/':
        return '_AGLO'
    elif enfoque == '/distintivo/':
        return ''
    else:
        print('El enfoque especificado no existe!!')

ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(ROOT_DIR, "datasets")
rt_path = programa + enfoque + programa +'-RT.txt'
k_fold_path = programa + enfoque + 'kfold_lists/orig/' + programa + name_file(enfoque) + '_ORIG_FOLD-0' + n + '_of_05-' + t + '.txt'

rt = os.path.join(DATA_DIR, rt_path)
kfold_list= os.path.join(DATA_DIR, k_fold_path)


def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            dataset.append(row.split("\t"))
    return dataset

def noDummies(dataset):
    for i in range(len(dataset)):
        dataset[i] = dataset[i][:-1]
    return dataset

def enterFilter(sentence):
    return sentence.replace('\n', '')

def datasetWithoutEnter(data):
    for row in data:
        row[2] = enterFilter(row[2])
    return data

def new_fold(all, k_fold_list):
    for row_all in all:
        for row_klist in k_fold_list:
            tweet_split = row_all[2].split(' ')
            tweet_interesting = ' '.join(tweet_split[2:])
            #print(tweet_interesting)
            #print(row_klist[2])
            if (row_klist[2] == tweet_interesting):
                k_fold_list.append(row_all)
                #print('true')
            else:
                #print('False')
                continue
    return k_fold_list

def list(data):
    list = []
    for row in data:
        row_split = row[2].split(' ')
        new_row = ' '.join(row_split[:-1])
        list.append(new_row)
    return list

def dataframe_to_compare(data):
    for row in data:
        tweet_split = row[2].split(' ')
        tweet_split[2] = tweet_split[2].replace(':', '')
        tweet_interesting = ' '.join(tweet_split)
        row[2] = tweet_interesting
    return data

def toPandas(data):
    return pd.DataFrame(data, columns= ["number", "label", "text"])

def compare(dataframe, list):
    dataframe['flag'] = dataframe.apply(lambda x: int(' '.join(x['text'].split(' ')[2:-1]) in list), axis=1)
    return dataframe

def join(df_flags, k_fold, rts):
    if (len(rts) != df_flags.shape[0]):
        print('Tamaño del dataset de RT y dataframe no coincide!!')
    else:
        for i in range(len(df_flags)):
            if(df_flags['flag'][i] == 1):
                k_fold.append(rts[i])
            else:
                continue
    return k_fold

def newFold(data):
    path = programa + enfoque + 'kfold_lists/complete/' + programa + name_file(enfoque) + '_ORIG_MAS_RT_FOLD-0' + n + '_of_05-' + t + '.txt'
    data.to_csv(os.path.join(
        DATA_DIR, path), index=False, header=None, sep='\t', doublequote=False)

def why_zeros(df_flags, rts):
    zeros = []
    if (len(rts) != df_flags.shape[0]):
        print('Tamaño del dataset de RT y dataframe no coincide!!')
    else:
        for i in range(len(df_flags)):
            if(df_flags['flag'][i] == 0):
                print(df_flags['text'][i])
                zeros.append(rts[i])
            else:
                continue
    return zeros

rt_data = noDummies(parseDataset(rt))
k_fold_data = datasetWithoutEnter(parseDataset(kfold_list))
print(len(k_fold_data))
print(len(rt_data))
tweets = list(k_fold_data)
data_to_dataframe = dataframe_to_compare(rt_data)
df_rt = toPandas(data_to_dataframe)
dataframe_with_flags = compare(df_rt, tweets)
final_list = join(dataframe_with_flags, k_fold_data, rt_data)
#zeros_list = why_zeros(dataframe_with_flags, rt_data)

print(dataframe_with_flags.groupby(['flag']).size())
df_final = toPandas(final_list)
newFold(df_final)