import os
programa = 'L6N_20151024'

ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(ROOT_DIR, "datasets")
dataset_path_orig = programa + '/distintivo/' + programa + '-ORIG.txt'
dataset_path_rt = programa + '/distintivo/' + programa + '-RT.txt'

dataset_file_orig = os.path.join(DATA_DIR, dataset_path_orig)
dataset_file_rt = os.path.join(DATA_DIR, dataset_path_rt)

def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            #añadimos cada fila al dataset pero haciendo un split por donde está la tabulacion
            dataset.append(row.split("\t"))
    return dataset
orig = parseDataset(dataset_file_orig)
rt = parseDataset(dataset_file_rt)
print(len(orig))
print(len(rt))