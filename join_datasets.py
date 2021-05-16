import os
import glob

enfoque = 'aglomerativo' #distintivo o aglomerativo
orig_o_rt = 'complete' #orig o complete
test_o_train = 'test' #test o train
n_fold = '1' #número del 1 al 5

ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(ROOT_DIR, "datasets")

files_path = 'L6N_ALL/' + enfoque + '/kfold_lists/' + orig_o_rt  + '/' + test_o_train + '/' + n_fold + '/' + '*.txt'
print(files_path)
path_input = os.path.join(DATA_DIR, files_path)
read_files = glob.glob(path_input)

name_enfoque = 'DIST' if enfoque == 'distintivo' else 'AGLO'
name_set = 'ORIG' if orig_o_rt == 'orig' else 'ORIG_MAS_RT'
name_type = 'TRAIN' if test_o_train == 'train' else 'TEST'

path = 'L6N_ALL/' + enfoque + '/kfold_lists/' + orig_o_rt  + '/' + 'L6N_ALL' + '_' + name_enfoque + '_' + name_set + '_FOLD-0'+ n_fold + '_of_05-' + name_type +'.txt'
path_output = os.path.join(DATA_DIR, path)
with open(path_output, "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())