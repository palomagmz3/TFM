
tweets_dataset = '/Users/palomagomez/PycharmProjects/TFM/bert_data/data_for_bert/L6N-20151128.txt'
bert_output = '/Users/palomagomez/PycharmProjects/TFM/bert_data/data_to_visualize/L6N20151128_cluster_20.csv'
tweets_with_label = '/Users/palomagomez/PycharmProjects/TFM/bert_data/data_with_label/L6N-20151128.txt'

from bert_code.topics_and_hashtags import L6N_20151128_20

def parseDataset(file):
    dataset = []
    try:
        with open(file, "r") as my_file:
            for row in my_file:
                #añadimos cada fila al dataset pero haciendo un split por donde está la tabulacion
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

def match(tweets, bert_output):
    data = []
    if (len(tweets) != len(bert_output)):
        print('El tamaño de los tweets y la salida de bert no coincide!!')
    else:
        for i in range(len(tweets)):
            list = []
            bert_output_split = bert_output[i].split(',')
            list.append(bert_output_split[0])
            list.append(tweets[i])
            data.append(list)
    return data

def get_labels(tweets, bert_output, n_topic):
    data = []
    if (len(tweets) != len(bert_output)):
        print('El tamaño de los tweets y la salida de bert no coincide!!')
    else:
        for i in range(len(tweets)):
            bert_output_split = bert_output[i].split(',')
            if (bert_output_split[0] == n_topic):
                data.append(tweets[i])
            else:
                continue
    return data

def bertopic_to_realhashtag(tweets, labels):
    for row in tweets:
        for topic in labels:
            if row[0] == topic:
                row[0] = labels[topic]
            #else:
            #    print(f'Ningún hashtag corresponde con el topic {row[0]}')
    return tweets

tweets_topics_bert = parseDataset(tweets_dataset)
bert_output = parseDataset(bert_output)
topics = topics(bert_output)
#print(topics)
together = match(tweets_topics_bert, bert_output)
tweets_bert_labels = bertopic_to_realhashtag(together, L6N_20151128_20)
#tweets_ntopic = get_labels(tweets, bert_output, '90')

'''

A partir de aquí comienzan los métodos para ver el acierto de las etiquetas dadas por bert
Esta parte corresponde a la evaluación

'''
from bert_code.data_for_BERT import parseDataset

tweets_real_labels = parseDataset(tweets_with_label)

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
