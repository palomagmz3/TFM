import os
import pandas as pd
import re

programa = 'L6N_20151107'
rt = False #True

ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(ROOT_DIR, "datasets")
dataset_path = programa + '/distintivo/' + programa +'-ALL.txt'
dataset_file = os.path.join(DATA_DIR, dataset_path)

def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            dataset.append(row.split("\t"))
    return dataset

def enterFilter(sentence):
    return sentence.replace('\n', '')

def datasetWithoutEnter(data):
    for row in data:
        row[3] = enterFilter(row[3])
    return data

def whatHappensToRT(data, rt=False):
    dataset = []
    if (rt==False):
        for row in data:
            tweet_split = row[2].split(' ')
            if (tweet_split[1] == 'RT'):
                continue
            elif (re.match('^@[A-Za-z0-9: @]+[\t ]+RT[\t ]+[A-Za-z0-9@_]+(RT)?[:]+', row[2])):
                continue
            else:
                dataset.append(row)
    else:
        for row in data:
            tweet_split = row[2].split(' ')
            if (tweet_split[1] == 'RT'):
                dataset.append(row)
            elif (re.match('^@[A-Za-z0-9: @]+[\t ]+RT[\t ]+[A-Za-z0-9@_]+(RT)?[:]+', row[2])):
                dataset.append(row)
            else:
                continue
    return dataset

def toPandas(data):
    return pd.DataFrame(data, columns= ["number", "label", "text", "name"])

def generateSplits(data, rt):
    label = 'ORIG' if rt == False else 'RT'
    path = programa + '/distintivo/' + programa + '-' + label + '.txt'
    data.to_csv(os.path.join(
        DATA_DIR, path), index=False, header=None, sep='\t', doublequote=False)


dataset = parseDataset(dataset_file)
print(len(dataset))
data_noEnter = datasetWithoutEnter(dataset)
data_manage_RT = whatHappensToRT(data_noEnter, rt)
df = toPandas(data_manage_RT)
print(df.shape[0])
df1 = df.drop_duplicates(subset='text', keep="first")
print(df1.shape[0])
generateSplits(df1, rt)

