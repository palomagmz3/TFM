import os
import pandas as pd
import csv
import sys
import seaborn as sns
import numpy as np

from only_labels import parseDataset, onlyTweetAndLabel, to_visualize, toPandas

program_name = 'L6N_ALL'
enfoque = 'distintivo' #aglomerativo o distintivo
num_topics = 110
embeddings = False #True o False

enfoque_name = 'AGLO' if enfoque=='aglomerativo' else 'DIST'
programa = 'L6N_ALL' + '_' + enfoque_name

e = '_embeddings' if embeddings==True else ''
n = '' if num_topics =='' else '_' + str(num_topics)
sys.path.append('../')

#paths to files: tweets (bert input), bert_output (topics and probs) and labels
cur_path = os.path.dirname(__file__)
tweets_file_path = '../bert_data/data_for_bert/' + programa +'.txt'
tweets_file = os.path.relpath(tweets_file_path, cur_path)
labels_path = '../datasets/' + program_name + '/' + enfoque + '/' + programa  + '_ORIG.txt'
labels_file = os.path.relpath(labels_path, cur_path)
bert_path = '../bert_data/data_from_bert_processed/' + programa  + e + n +'.csv'
bert_file = os.path.relpath(bert_path, cur_path)

labels = onlyTweetAndLabel(parseDataset(labels_file))

def parseDataset(file):
    dataset = []
    try:
        with open(file, "r") as my_file:
            for row in my_file:
                row = row.replace('\n', '')
                dataset.append(row)
    except:
        print('No se puede leer el archivo')
    return dataset

bert_output = parseDataset(bert_file)
toPandas(to_visualize(bert_output, labels), programa, n, e)

def topics(bert_output):
    list = []
    for row in bert_output:
        bert_output_split = row.split(',')
        if list.__contains__(bert_output_split[0]):
            continue
        else:
            list.append(bert_output_split[0])
    return list

def join (hashtags, bert_output):
    # Salida es un dataset con tres columnas: topic de bert, etiqueta original y tweet
    data = []
    if (len(hashtags) != len(bert_output)):
        print('El tamaño de los tweets y la salida de bert no coincide!!')
    else:
        for i in range(len(hashtags)):
            list = []
            bert_output_split = bert_output[i].split(',')
            list.append(bert_output_split[0])
            list.append(hashtags[i])
            data.append(list)
    return data

def get_tweets_for_each_topic(data, topic):
    data_aux = []
    for row in data:
        if row[0] == topic:
            data_aux.append(row)
        else:
            continue
    return data_aux

def get_max_label(data):
    dict = {}
    for row in data:
        if (row[1] in dict):
            dict[row[1]] += 1
        else:
            dict[row[1]] = 1
    max_key = max(dict, key=lambda key: dict[key])
    #print(dict)
    return max_key

def automatch(data, topics):
    # Método para etiquetar los tweets de bert según los topics
    # Utiliza los dos métodos anteriores
    for topic in topics:
        data_aux = get_tweets_for_each_topic(data, topic)
        max_key = get_max_label(data_aux)
        for row in data:
            if row[0] == topic:
                row[0] = max_key
                del row[1]
            else:
                continue
    return data

def to_visualize(bert_data, labeled_data):
    dataset = []
    # Salida es lo mismo que sale de Bert pero la primera columna ya no es un número de topic sino un hashtag
    # Usamos el método tanto para la etiqueta nueva de Bert como para la original
    if (len(bert_data) != len(labeled_data)):
        print('El tamaño de los tweets y la salida de bert no coincide!!')
    else:
        for i in range(len(labeled_data)):
            list = []
            list.append(labeled_data[i][0])
            new_row = labeled_data[i][0] + ',' + ','.join(bert_data[i].split(',')[1:])
            n = new_row.split(',')
            if(n[0][-1].isdigit() == True):
                n[0] = n[0][:-1]
            new_row = ','.join(n)
            bert_data[i] = new_row
    return bert_data

def toCSV(data):
    file_path = '../bert_data/data_to_visualize/' + programa + e + n + '-bert_labels.csv'
    df = pd.DataFrame(data)
    df.to_csv(os.path.relpath(file_path, cur_path), index=False, header=None, sep='\t', doublequote=False, quoting=csv.QUOTE_NONE)


bert_output = parseDataset(bert_file)
topics = topics(bert_output)
tweets_bert_and_labels = join(labels, bert_output)
#a = get_tweets_for_each_topic(tweets_bert_and_labels, '-1')

tweets_bert_labels = automatch(tweets_bert_and_labels, topics)

bert_visualize = to_visualize(bert_output, tweets_bert_labels)
toCSV(bert_visualize)



'''

A partir de aquí comienzan los métodos para ver el acierto de las etiquetas dadas por bert
Esta parte corresponde a la evaluación

'''

def results(tweets_original_labels, tweets_after_bert):
    aciertos = 0
    fallos = 0
    if (len(tweets_original_labels) != len(tweets_after_bert)):
        print('El tamaño de los tweets y la salida de bert no coincide!!')
    else:
        for i in range(len(tweets_original_labels)):
            #print(tweets_original_labels[i][0] + 'y' + tweets_after_bert[i][0])
            if tweets_original_labels[i][0] == tweets_after_bert[i][0]:
                aciertos += 1
            else:
                fallos +=1
    print(f'aciertos: {aciertos} fallos: {fallos} de un dataset de longitud: {len(tweets_original_labels)}')

def get_categories(labels):
    categories = []
    for row in labels:
        if row in categories:
            continue
        else:
            categories.append(row)
    return categories

def getY(tweets):
    y = []
    for row in tweets:
        y.append(row[0])
    return y

y_expected = labels

y_predicted = tweets_bert_labels
categories = get_categories(labels)

from sklearn import metrics
import matplotlib.pyplot as plt


results(labels, tweets_bert_labels)
#print(f'Accuracy = {metrics.accuracy_score(y_expected, y_predicted)}')
print(metrics.classification_report(y_expected, y_predicted))
cf_matrix = metrics.confusion_matrix(y_expected, y_predicted)
print(cf_matrix)
file_path = '../bert_data/data_to_visualize/' + programa + e + n +'_cfmatrix.csv'
pd.DataFrame(cf_matrix).to_csv(os.path.relpath(file_path, cur_path))
ax = plt.subplot()
sns.set(font_scale=0.8)
sns.heatmap(cf_matrix, annot=True, xticklabels=categories, yticklabels=categories, fmt='g')
ax.tick_params(axis='both', which='major', labelsize=10)
plt.show()

#sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, xticklabels=categories, yticklabels=categories,
            #fmt='.2%', cmap='YlGnBu')
#plt.show()

