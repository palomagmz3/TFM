import os
import glob

enfoque = 'aglomerativo' #distintivo
orig_o_rt = 'ORIG' #RT

ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(ROOT_DIR, "datasets")

files_path = 'L6N_ALL/' + enfoque + '/*' + orig_o_rt  + '.txt'
path_input = os.path.join(DATA_DIR, files_path)
read_files = glob.glob(path_input)

name = 'DIST' if enfoque == 'distintivo' else 'AGLO'
path = 'L6N_ALL/' + enfoque + '/L6N_ALL' + '_' + name + '_' + orig_o_rt + '.txt'
path_output = os.path.join(DATA_DIR, path)
with open(path_output, "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())