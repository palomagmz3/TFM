import re

dataset_file = '../bert_data/data_from_bert/L6N20151128_11topics.txt'

def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            #añadimos cada fila al dataset pero haciendo un split por donde está la tabulacion
            row = row.replace('\n', '')
            dataset.append(row)
    return dataset

def getWords(data):
    dataset = []
    for row in data:
        row = re.sub(r'[\[\]\(\):,]', r'', row)
        dataset.append(row.split(' ')[0])
        row_split = row.split(' ')[1::2]
        dataset.append(', '.join(row_split))
    return dataset

data = parseDataset(dataset_file)
for row in getWords(data):
    print(row)