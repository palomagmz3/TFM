

'''
Este archivo de python es de pruebas para leer líneas de datasets y comprobar que sigen la misma estructura que los
training/validation/test que creó Ricardo con su código a partir del csv COVID-DEMO.csv
'''

dataset_l6n = 'datasets/L6N_20151024/training.txt'
dataset_covid = 'datasets/COVID-DEMO/training.txt'

def readL6N (file):
    with open(file, "r") as my_file:
        print(my_file.readline())
        print(my_file.readline())
        print(my_file.readline())
        print(my_file.readline())
        lineL6N = my_file.readline()
        print(lineL6N.strip().split('\t'))

def readCovid (file):
    with open(file, "r") as my_file:
        print(my_file.readline())
        print(my_file.readline())
        print(my_file.readline())
        print(my_file.readline())
        lineCovid = my_file.readline()
        print(lineCovid.strip().split('\t'))

readCovid(dataset_covid)
readL6N(dataset_l6n)

#with open(join(DATA_DIR, dataset_name, key + '.txt'), 'r') as file:
 #   data = [line.strip().split('\t') for line in file.readlines()]
#X = [d[2] for d in data]
#y = [d[1] for d in data]
#names = [d[3] for d in data]