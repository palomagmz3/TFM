import os
import glob

ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(ROOT_DIR, "datasets")
path_input = os.path.join(DATA_DIR, "organizativo/all/*.txt")
read_files = glob.glob(path_input)

path_output = os.path.join(DATA_DIR, "organizativo/all/ALL.txt")
with open(path_output, "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())