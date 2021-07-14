import re
import pandas as pd
import os
import sys

#programa = 'L6N_20151024'
sys.path.append('../')
cur_path = os.path.dirname(__file__)

def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            dataset.append(row.split("\t"))
    return dataset

def onlyTweetAndLabel(data):
    dataset = []
    for row in data:
        if (row[1] == '#L6Nanticorrupción' or row[1] == '#L6Ncorruptelas' or row[1] == '#L6Ncorruptos' or row[1] == '#L6Nyoconfieso'):
            label = '#L6Ncorrupción'
            dataset.append(label)
        elif (row[1] == '#L6Nantiyihadismo' or row[1] == '#L6Nmachismo' or row[1] == '#L6Nataquekabul'):
            label = '#L6Njesuisparis'
            dataset.append(label)
        elif (row[1] == '#L6Nayudasocial' or row[1] == '#L6Nretoempleo' or row[1] == '#L6Nretoeduación' or row[1] == '#L6Nsueldos' or row[1] == '#L6Nsinluz' or row[1] == '#L6Npizarraempleo' or row[1] == '#L6Npizarrasparo' or row[1] == '#L6Npizarrapensiones' or row[1] == '#L6Nprecioluz'):
            label = '#L6Nreto'
            dataset.append(label)
        elif (row[1] == '#L6Nbarómetro' or row[1] == '#L6Nbunbury' or row[1] == '#L6Nvotorogado' or row[1] == '#L6Npactómetro' or row[1] == '#L6Nresacaelectoral' or row[1] == '#L6Ncampaña20D' or row[1] == '#L6Nencampaña' or row[1] == '#L6Nobjetivo20D' or row[1] == '#L6Ntictacpactos' or row[1] == '#L6Npresidudas' or row[1] == '#L6Nnoupresident' or row[1] == '#L6Npablopineda' or row[1] == '#L6Nalerta' or row[1] == '#L6Ncongreso' or row[1] =='#L6Ncuentatrás20D' or row[1] == '#L6Nlíopactos'):
            label = '#L6Nelecciones'
            dataset.append(label)
        elif (row[1] == '#L6Niglesias24h'):
            label = '#L6Ncalleiglesias'
            dataset.append(label)
        elif (row[1] == '#L6Nrivera24h'):
            label = '#L6Ncallerivera'
            dataset.append(label)
        elif (row[1] == '#L6Nsánchez24h'):
            label = '#L6Ncallepsánchez'
            dataset.append(label)
        elif (row[1] == '#L6Ncuptalunya' or row[1] == '#L6Ndesafíocat' or row[1] == '#L6Nelclanpujol' or row[1] == '#L6Nmasmordidas' or row[1] == '#L6Nretocatalán'):
            label = '#L6Ncataluña'
            dataset.append(label)
        elif (row[1] == '#L6NfranKO'):
            label = '#L6Nfranquismo'
            dataset.append(label)
        else:
            dataset.append(row[1])
    return dataset

def to_visualize(bert_data, labeled_data):
    # Salida es lo mismo que sale de Bert pero la primera columna ya no es un número de topic sino un hashtag
    # Usamos el método tanto para la etiqueta nueva de Bert como para la original
    if (len(bert_data) != len(labeled_data)):
        print('El tamaño de los tweets y la salida de bert no coincide!!')
    else:
        for i in range(len(labeled_data)):
            new_row = labeled_data[i] + ',' + ','.join(bert_data[i].split(',')[1:])
            bert_data[i] = new_row
    return bert_data

def toPandas(data, programa, n, e):
    file_path = '../bert_data/data_to_visualize/' + programa + e + n + '-orig_labels.csv'
    #file_path = '../bert_data/data_for_bert/' + programa + '_'+ name_enfoque +'-orig_labels.txt'
    data_to_pandas = pd.DataFrame(data)
    data_to_pandas.to_csv(os.path.relpath(file_path, cur_path), index=False, header=None, sep='\t', doublequote=False)
