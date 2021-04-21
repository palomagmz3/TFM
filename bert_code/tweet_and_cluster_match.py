import os
import pandas as pd
import csv

ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(ROOT_DIR, "bert_data")

tweets_dataset = os.path.join(DATA_DIR, "data_for_bert/L6N-20151128.txt")
bert_output = os.path.join(DATA_DIR, "data_from_bert_processed/L6N20151128_cluster_20.csv")
tweets_with_label = os.path.join(DATA_DIR, "data_with_label/L6N-20151128.txt")


def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            dataset.append(row.split("\t"))
    return dataset

tweets_real_labels = parseDataset(tweets_with_label)

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
            list.append(hashtags[i][0])
            list.append(hashtags[i][1])
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
            new_row = labeled_data[i][0] + bert_data[i][1:]
            n = new_row.split(',')
            if(n[0][-1].isdigit() == True):
                n[0] = n[0][:-1]
            new_row = ','.join(n)
            bert_data[i] = new_row
    return bert_data

def toCSV(data):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_DIR, 'data_to_visualize/L6N20151128_cluster_20_labels.csv'), index=False, header=None, sep='\t', doublequote=False, quoting=csv.QUOTE_NONE)

tweets_topics_bert = parseDataset(tweets_dataset)
bert_output = parseDataset(bert_output)
topics = topics(bert_output)
tweets_bert_and_labels = join(tweets_real_labels, bert_output)
tweets_bert_labels = automatch(tweets_bert_and_labels, topics)

'''
IMPORTANTE: cada vez que se utilice este script para un programa hay que ejecutar las 4 líneas de abajo de dos en dos:
primero las dos primeras y se comentan y luego las dos siguientes. TIENE QUE HABER UN PAR DE LAS LÍNEAS SIEMPRE COMENTADO
'''
#bert_visualize = to_visualize(bert_output, tweets_bert_labels)
#toCSV(bert_visualize)

labels_visualize = to_visualize(bert_output, tweets_real_labels)
toCSV(labels_visualize)



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


def getY(tweets):
    y = []
    for row in tweets:
        y.append(row[0])
    return y

y_expected = getY(tweets_real_labels)
y_predicted = getY(tweets_bert_labels)
from sklearn import metrics

results(tweets_real_labels, tweets_bert_labels)
print(f'Accuracy = {metrics.accuracy_score(y_expected, y_predicted)}')
print(metrics.classification_report(y_expected, y_predicted))
print(metrics.confusion_matrix(y_expected, y_predicted))
