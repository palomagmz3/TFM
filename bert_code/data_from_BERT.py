import re
import csv
import pandas as pd
import os

#Importamos data que salga de aplicar Bert
dataset_file = '../bert_data/data_from_bert/L6N20151128_11topics.txt'

def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            #añadimos cada fila al dataset pero haciendo un split por donde está la tabulacion
            row = row.replace('\n', '')
            dataset.append(row)
    return dataset

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
    df = pd.DataFrame(data, columns=['text'])
    df.to_csv(os.path.join('../bert_data/data_to_visualize/L6N20151128_cluster_20.csv'), index=False, header=None, sep='\t', doublequote=False, quoting=csv.QUOTE_NONE)

data = parseDataset(dataset_file)
data_for_visualize = processMatrix(data)
toCSV(data_for_visualize)