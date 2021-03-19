import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PATH, "datasets")

dataset_file= '../programas/L6N-20151024.txt'
dataset_file2='../programas/L6N-20151031.txt'
#dataset_file_small='L6N-20151024-1500lines.txt'
pandas_csv = 'out.csv'

#Convertir dataset en array de frases
def dataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            dataset.append(row)
    return dataset

#Quitar fecha y hora a una frase del dataset (solo dejamos id, hashtag y tweet)
def datetimeRemovalAux(sentence):
    dateRemoved = re.sub(r'(\d{4})(-)(\d{1,2})(-)(\d{1,2})', r'', sentence)
    timeRemoved = re.sub(r'(\d{1,2})(:)(\d{1,2}):(\d{1,2})', r'', dateRemoved)
    print(timeRemoved)


#print(string2.split("\t"))

"""
El programa empieza aquí, lo anterior son pruebas
Hay varias partes en el siguiente orden:
1) Parsear el dataset (método parseDataset) donde creamos una lista formada por listas/arrays que van a ser las filas del documento que se pasa como parámetro.
   Cada lista/array estará inicialmente formada por cinco campos: id del tweet, fecha y hora, etiqueta (hashtag), usuario que mandó el tweet y el contenido del tweet.
2) Borrar el segundo campo fecha y hora (datetimeRemoval). Si lo quisiésemos conservar obviaríamos este método
3) Añadir etiqueta a los tweets que no tienen hashtag (addLabel). Este método añade un hashtag cogido del cuerpo del tweet (en caso de tratarse de un programa sin ninguna etiqueta
   Si solo faltan algunas etiquetas se utilizará mejor el método addHashtag que pone el hashtag #NoHashtag a los tweets sin hashtag
   *La etiqueta se añade sobre el segundo campo en caso de haber quitado la fecha y hora. Si no la hemos quitado habrá que modificar el método.
4) Fusionar los campos del usuario y del tweet (mergeUserTweet), ya que para entrenar el modelo solo queremos campo id, etiqueta y texto.
   Otra opción sería eliminar este campo (userRemoval) 
5) Quitar el hashtag de dentro de tweet, si está (hashtagFilter) 
6) Quitar las comillas del contenido del tweet si se desea (quotationMarksFilter)
7) Una vez tenemos el contenido con el formato deseado (sin fecha, sin hashtag en el tweet, con tres columnas, etc) creamos el dataframe con pangas (toPandas)
8) A partir del dataframe de pandas, creamos los training/validation/test sets (generateSplits)     
"""

def parseDataset(file):
    dataset = []
    with open(file, "r") as my_file:
        for row in my_file:
            #añadimos cada fila al dataset pero haciendo un split por donde está la tabulacion
            dataset.append(row.split("\t"))
    return dataset

#print(parseDataset(dataset_file_aux))
parse = parseDataset(dataset_file)
parse2 = parseDataset(dataset_file2)

def datetimeRemoval(data):
    for row in data:
        del row[1]
    return data

#print(datetimeRemoval(parse))
dateRemoved = datetimeRemoval(parse)
dateRemoved2 = datetimeRemoval(parse2)

def gapsRemoval(sentence):
    for item in sentence:
        if (item == ''):
            sentence.remove(item)
    return sentence

#En caso de que el dataset no tenga campo de etiqueta
def addLabel(data):
    for row in data:
        row_one_space = re.sub('\s+',' ',row[3])
        row_split = row_one_space.split(' ')
        new_row = gapsRemoval(row_split)
        label = []
        for item in new_row:
            if (item[0]=='#'):
                label.append(item)
        if (len(label)==0):
            row[1] = '#NoHashtag'
            #row.insert(1, '#NoHashtag')
        else:
            row[1] = ','.join(label)
            #row.insert(1, ','.join(label))
    return data

dataWithLabel = addLabel(dateRemoved2)

def addHashtag(data):
    for row in data:
        #Row[1] si hemos hecho antes el dateRemoved, si no cambiar a row[2]
        if (row[1]==''):
            row[1] = '#NoHashtag'
    return data

dataLabeled = addHashtag(dateRemoved)

#lista = ['614761947976835072', '#L6Ncasado', '@InfoLordfit', '#L6Ncasado Hace tiempo que sabemos perfectamente, que lo que habláis, es Lo contrario, de lo que haréis.\n']
#lista[2:4] = [' '.join(lista[2:4])]
#print(lista)

def mergeUserTweet(data):
    for row in data:
        row[2:4] = [' '.join(row[2:4])]
    return data

dataMerged = mergeUserTweet(dataLabeled)
dataMerged2 = mergeUserTweet(dataWithLabel)
#Si en lugar de unir los campos usuario y tweet queremos eliminar el campo usuario
def userRemoval(data):
    for row in data:
        del row[2]
    return data

lista = '@martalatita #L6Nalcaldías #L6Ncasado Los que meten miedo con dictaduras bolivarianas son los que defienden la DICTADURA DE FRANCO'
lista = re.sub(r'(@[A-Za-z0-9\_áéíóúüñÁÉÍÓÚÜÑ]+)', r'', lista)
print(lista)

def hashtagFilter(sentence):
    return re.sub(r'(#[A-Za-z0-9\_áéíóúüñÁÉÍÓÚÜÑ]+)', r'', sentence)

def datasetWithoutHashtag(data):
    for row in data:
        row[2] = hashtagFilter(row[2])
    return data

dataFiltered = datasetWithoutHashtag(dataMerged)
data2Filtered = datasetWithoutHashtag(dataMerged2)

def quotationMarksFilter(sentence):
    #return re.sub(r'', r'', sentence)
    return sentence.replace('"', '');

#print(quotationMarksFilter("@copitoypata RT @podemosmad: ""Los concejales de Podemos son herederos de ese maravilloso espíritu del 15M"" @ManuelaCarmena  https://t.co/bjtB…)"))

def datasetWithoutQuotationMarks(data):
    for row in data:
        row[2] = quotationMarksFilter(row[2])
    return data

dataFiltered2 = datasetWithoutQuotationMarks(dataFiltered)
data2Filtered2 = datasetWithoutQuotationMarks(data2Filtered)
#print(hashtagFilter(dataMerged[1][2]))

def enterFilter(sentence):
    return sentence.replace('\n', '')

def datasetWithoutEnter(data):
    for row in data:
        row[2] = enterFilter(row[2])
    return data

dataFiltered3 = datasetWithoutEnter(dataFiltered2)
data2Filtered3 = datasetWithoutEnter(data2Filtered2)

#Si queremos eliminar todo lo que tenga arrobas - de momento no
def arrobaFilter(sentence):
    return re.sub(r'(@[A-Za-z0-9\_áéíóúüñÁÉÍÓÚÜÑ]+)', r'', sentence)

def datasetWithoutArroba(data):
    for row in data:
        row[2] = arrobaFilter(row[2])
    return data

def multilabelFilter(label):
    nAlmohadilla = 0
    for i in label:
        if (i=='#'):
            nAlmohadilla += 1
    if (nAlmohadilla>1):
        return True
    else:
        return False

def datasetWithoutMultilabel(data):
    for row in data:
        if (multilabelFilter(row[1])):
            data.remove(row)
    return data

dataFiltered4 = datasetWithoutMultilabel(dataFiltered3)

#for row in dataFiltered4:
#    print(row)
"""
Una vez procesado el archivo podemos convertirlo a un dataframe de pandas.
Hacemos splits para crear los sets train/test --> depende del modelo que vayamos a usar (modelo de fernando solo usa train y test
mientras que el modelo de Ricardo usa también validation

"""
def toPandas(data):
    #data = pandas.read_csv(dataset_file_aux, sep='\t', header=None)
    return pd.DataFrame(data, columns= ["number", "label", "text"])

#print(toPandas(dataFiltered3))
df = toPandas(dataFiltered4)
df2 = toPandas(data2Filtered3)

print(df['label'])
print(df2['label'])

def generateSplits(data):
    #Anadimos cuarta columna (el modelo del laboratorio la utiliza)
    data["name"] = ["dummy_name" + str(i) for i in range(len(data))]
    # Hacemos set de train y test con función importada antes. Testset = 30% del total
    train, test = train_test_split(data, test_size=0.3)
    # A partir del train set sacamos el propio train set y el validation set. Validationset = 15%
    #train, validation = train_test_split(train, test_size=0.15)

    # Guardamos los sets train/validation/test creados mediante pandas.
    # DATA_DIR = 'datasets/' (carpeta datasets)
    # IMPORTANTE: guardar estos datasets con ESTOS NOMBRES
    data.to_csv(os.path.join(
        '/Users/palomagomez/PycharmProjects/TFM/datasets', 'L6N-20151024', 'all.txt'), index=False, header=None, sep='\t', doublequote=False)
    train.to_csv(os.path.join(
        '/Users/palomagomez/PycharmProjects/TFM/datasets', 'L6N-20151024', 'training.txt'), index=False, header=None, sep='\t', doublequote=False)
    #validation.to_csv(os.path.join(
        #DATA_DIR, 'L6N-20151024', 'validation.txt'), index=False, header=None, sep='\t', doublequote=False)
    test.to_csv(os.path.join(
        '/Users/palomagomez/PycharmProjects/TFM/datasets', 'L6N-20151024', 'test.txt'), index=False, header=None, sep='\t')

generateSplits(df)

def generateSplits2(data):
    #Anadimos cuarta columna (el modelo del laboratorio la utiliza)
    data["name"] = ["dummy_name" + str(i) for i in range(len(data))]
    # Hacemos set de train y test con función importada antes. Testset = 30% del total
    train, test = train_test_split(data, test_size=0.3)
    # A partir del train set sacamos el propio train set y el validation set. Validationset = 15%
    #train, validation = train_test_split(train, test_size=0.15)

    # Guardamos los sets train/validation/test creados mediante pandas.
    # DATA_DIR = 'datasets/' (carpeta datasets)
    # IMPORTANTE: guardar estos datasets con ESTOS NOMBRES
    data.to_csv(os.path.join(
        '/Users/palomagomez/PycharmProjects/TFM/datasets', 'L6N-20151031', 'all.txt'), index=False, header=None, sep='\t', doublequote=False)
    train.to_csv(os.path.join(
        '/Users/palomagomez/PycharmProjects/TFM/datasets', 'L6N-20151031', 'training.txt'), index=False, header=None, sep='\t', doublequote=False)
    #validation.to_csv(os.path.join(
        #DATA_DIR, 'L6N-20151024', 'validation.txt'), index=False, header=None, sep='\t', doublequote=False)
    test.to_csv(os.path.join(
        '/Users/palomagomez/PycharmProjects/TFM/datasets', 'L6N-20151031', 'test.txt'), index=False, header=None, sep='\t')

generateSplits2(df2)
