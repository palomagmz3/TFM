import os
import glob
import sys
import pandas as pd

sys.path.append('../')
cur_path = os.path.dirname(__file__)

def join_dist_pairs():
    files_path = '../bert_data/data_for_embeddings/' + '*-dist_pairs.txt'

    files = os.path.relpath(files_path, cur_path)
    read_files = glob.glob(files)

    path = '../bert_data/data_for_embeddings/final/' + 'L6N_ALL-dist_pairs.txt'
    path_output = os.path.relpath(path, cur_path)
    with open(path_output, "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())

# Abrimos el archivo con las parejas que no son iguales y nos quedamos con la misma cantidad que de parejas iguales
def resize_dist_pairs_set():
    file_dist_pairs_path  = '../bert_data/data_for_embeddings/final/' + 'L6N_ALL-dist_pairs.txt'
    file_dist_pairs = os.path.relpath(file_dist_pairs_path, cur_path)
    dataset = []
    with open(file_dist_pairs, "r") as my_file:
        for row in my_file:
            dataset.append(row.split("\t"))
    for row in dataset:
        row[4] = row[4].replace('\n', '')
    df = pd.DataFrame(dataset, columns= ["hashtag1", "hashtag2", "tweet1", "tweet2", "label"])
    df = df.sample(n=70676)
    df.to_csv(os.path.relpath(file_dist_pairs_path, cur_path), index=False, header=None, sep='\t', doublequote=False)

# Juntamos los dos archivos: parejas iguales y parejas distintas
def last_join():
    files_path = '../bert_data/data_for_embeddings/final/' + '*.txt'

    files = os.path.relpath(files_path, cur_path)
    read_files = glob.glob(files)

    path = '../bert_data/data_for_embeddings/final/' + 'L6N_ALL-all_pairs.txt'
    path_output = os.path.relpath(path, cur_path)
    with open(path_output, "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())

join_dist_pairs()
resize_dist_pairs_set()
last_join()
