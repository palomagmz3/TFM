from dataloaders.task1 import parse as parse1
from predict.predictions import dump_attentions
from utils.train import load_pretrained_model
from dataloaders.baseline_dataloader import parse
from logger.training import LabelTransformer
from utils.train import load_embeddings, get_pipeline
from utils.nlp import twitter_preprocess
from config import DEVICE, BASE_PATH
from modules.nn.dataloading import WordDataset
from torch.utils.data import DataLoader
from logger.training import predict
import numpy
import os
import json
import sys
from torch.nn import Softmax
from torch import FloatTensor

from typing import NamedTuple


#class User(NamedTuple):
#    name: str

class confusion_matrix_struct(NamedTuple):
    num_items_per_class: list
    avg_length_per_class: list
    num_correct: int
    avg_length_correct: float
    num_errors: int
    avg_length_errors: float
    accuracy: float
    num_total_samples: int
    #qux: User

PREDICTIONS_DIR = os.path.join(BASE_PATH, 'out')
ROOT_DIR = os.path.abspath(os.curdir)

L6N_folder= ROOT_DIR + '/'
L6N_topics_file='L6N_DIST_topics.txt'
task = 34 # "COVID_entities"
model_names = ["L6N_ALL_DIST_ORIG_ENC-100_FOLD-01_of_05_0.1545",
    "L6N_ALL_DIST_ORIG_ENC-100_FOLD-02_of_05_0.1544",
    "L6N_ALL_DIST_ORIG_ENC-100_FOLD-03_of_05_0.1554",
    "L6N_ALL_DIST_ORIG_ENC-100_FOLD-04_of_05_0.1482",
    "L6N_ALL_DIST_ORIG_ENC-100_FOLD-05_of_05_0.1488"]

dataset_keys = ['test_fold-1',
                'test_fold-2',
                'test_fold-3',
                'test_fold-4',
                'test_fold-5']

name = 'L6N_ALL_DIST'

def read_topics_COVID(filename):
    topics = []
    f = open(filename, "r")
    lines = f.readlines()
    for x in lines:
        topics.append(x.rstrip('\n'))
    f.close()
    return topics

def predict_COVID(task=34, model_name="FFM_demo_0.7606", dataset_key="", label_set=[]):
    model, config = load_pretrained_model(model_name)

    #print(label_set)
    # pass a transformer function, for preparing tha labels for training
    label_map = {label: idx for idx, label in
                 enumerate(sorted(list(set(label_set))))}
    inv_label_map = {v: k for k, v in label_map.items()}
    transformer = LabelTransformer(label_map, inv_label_map)

    word2idx, idx2word, embeddings = load_embeddings(config)

    # dummy scores if order to utilize Dataset classes as they are
    #dummy_y = [0] * len(data)

    preprocessor = twitter_preprocess()

    X, y = parse(task=task, dataset=dataset_key)

    model.to(DEVICE)
    task = "clf"
    pipeline = get_pipeline(task=task, eval=True)

    dataset = WordDataset(X, #data[0],  # data,
                          y, #data[1],  # dummy_y,
                          word2idx,
                          #max_length=50,
                          name=None,  # name=name,
                          preprocess=preprocessor,
                          verbose=False,
                          label_transformer=transformer)

    batch_size = 64#2
    loader = DataLoader(dataset, batch_size)

    avg_loss, (dummy_y, pred), posteriors, attentions = predict(model,
                                                                pipeline,
                                                                loader,
                                                                task,
                                                                "eval")

    tokens = loader.dataset.data

    data = []
    for tweet, label, prediction, posterior, attention in zip(tokens, y,
                                                              pred, posteriors,
                                                              attentions):
        label = numpy.array(label)
        #prediction = numpy.array(prediction).astype(label.dtype)
        prediction = numpy.array(prediction)

        posterior_parser = Softmax(dim=0)
        posteriors_tensor = FloatTensor(posterior)
        posteriors_parsed = posterior_parser(posteriors_tensor)
        posteriors_list = posteriors_parsed.tolist()

        item = {
            "text": tweet,
            "label": label.tolist(),
            "prediction": prediction.tolist(),
            #"posterior (raw)": numpy.array(posterior).tolist(),
            #"posterior (softmax)": posteriors_list,
            #"attention": numpy.array(attention).tolist(),
        }

        #print_ECI_sentiment_results (item, label_map, attention=True)
        #print_ECI_sentiment_results(item, label_map, attention=False)


        data.append(item)
    with open(os.path.join(PREDICTIONS_DIR, "{}.json".format(name+'_'+dataset_key)), 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))

    return data

def GetConfusionMatrix(test_data, classes):
    num_items_test = len(test_data)
    num_classes = len(classes)
    print('Creating a confusion matrix...')
    print('%d test items for %d classes...' % (num_items_test, num_classes))

    # Create a list containing num_classes lists, each of num_classes items, all set to 0
    w, h = num_classes, num_classes;
    matrix = [[0 for x in range(w)] for y in range(h)]
    error_matrix = [[0 for x in range(w)] for y in range(h)]

    # Also measures average sentence length for each class
    avg_length_per_class = [0 for x in range(num_classes)]
    num_items_per_class = [0 for x in range(num_classes)]
    # ...and average length per result
    avg_length_correct = 0
    num_correct = 0
    avg_length_errors = 0
    num_errors = 0

    num_items_test = 0
    # Fill the matrix with the results
    for item in test_data:
        try:
            label_coord = classes.index(item['label'])
        except (IndexError, ValueError):
            print('EXCEPTION!!!!!!')

        try:
            prediction_coord = classes.index(item['prediction'])
        except (IndexError, ValueError):
            print('EXCEPTION!!!!!!')

        if label_coord < 0 or label_coord > num_classes:
            print('PROBLEMS!!!! label_coord = %d!!!!' % label_coord)
            break
        if prediction_coord < 0 or prediction_coord > num_classes:
            print('PROBLEMS!!!! prediction_coord = %d!!!!' % prediction_coord)
            break

        num_items_per_class[label_coord] += 1
        avg_length_per_class[label_coord] += len(item['text'])

        if label_coord == prediction_coord:
            avg_length_correct += len(item['text'])
            num_correct += 1
        else:
            avg_length_errors += len(item['text'])
            num_errors += 1

        matrix[label_coord][prediction_coord] = matrix[label_coord][prediction_coord] + 1
        if label_coord != prediction_coord:
            error_matrix[label_coord][prediction_coord] = error_matrix[label_coord][prediction_coord] + 1
        num_items_test += 1

    for item in range(num_classes):
        if num_items_per_class[item] > 0:
            avg_length_per_class[item] = avg_length_per_class[item] / num_items_per_class[item]
        else:
            avg_length_per_class[item] = 0

    if num_correct > 0:
        avg_length_correct = avg_length_correct / num_correct
    else:
        avg_length_correct = 0

    if num_errors > 0:
        avg_length_errors = avg_length_errors / num_errors
    else:
        avg_length_errors = 0

    if (num_correct + num_errors) != num_items_test:
        print('PROBLEMS!!!! num_correct(%d) + num_errors(%d) != num_items_test(%d)!!!!' % (num_correct, num_errors, num_items_test))
        exit()

    if num_items_test > 0:
        accuracy = 100 * num_correct / num_items_test
    else:
        accuracy = 0

    statistics = confusion_matrix_struct(num_items_per_class, avg_length_per_class, num_correct, avg_length_correct, num_errors, avg_length_errors, accuracy, num_items_test)

    return matrix, statistics, error_matrix

def PrintStatistics(filename, statistics, classes):
    # class confusion_matrix_struct(NamedTuple):
    keys = {'num_items_per_class': 0,
            'avg_length_per_class': 1,
            'num_correct': 2,
            'avg_length_correct': 3,
            'num_errors': 4,
            'avg_length_errors': 5,
            'accuracy': 6,
            'num_total_samples': 7
            }

    with open(filename, 'w') as f:
        num_classes = len(classes)
        print('\n[TEST STATISTICS]\n')
        print('\t[#][CLASS][# SAMPLES][AVG LENGTH]\n')

        f.write('\n[TEST STATISTICS]\n')
        f.write('#\tCLASS\t# SAMPLES\tAVG LENGTH\n')

        for i in range(num_classes):
            print('\t[%d][%s][%d][%2.2f]' % (i, classes[i], statistics[keys['num_items_per_class']][i], statistics[keys['avg_length_per_class']][i]))
            f.write('%d\t%s\t%d\t%2.2f\n' % (i, classes[i], statistics[keys['num_items_per_class']][i], statistics[keys['avg_length_per_class']][i]))

        print('\n\t[SUMMARY]')
        print('\t- NUM TOTAL SAMPLES = %d' % statistics[keys['num_total_samples']])
        print('\t- NUM SAMPLES CORRECT = %d' % statistics[keys['num_correct']])
        print('\t- NUM SAMPLES WRONG = %d' % statistics[keys['num_errors']])
        print('\t- ACCURACY = %2.2f' % statistics[keys['accuracy']])
        print('\t- AVG LENGTH CORRECT = %2.2f' % statistics[keys['avg_length_correct']])
        print('\t- AVG LENGTH WRONG = %2.2f' % statistics[keys['avg_length_errors']])

        f.write('\n[SUMMARY]\n')
        f.write('- NUM TOTAL SAMPLES = %d\n' % statistics[keys['num_total_samples']])
        f.write('- NUM SAMPLES CORRECT = %d\n' % statistics[keys['num_correct']])
        f.write('- NUM SAMPLES WRONG = %d\n' % statistics[keys['num_errors']])
        f.write('- ACCURACY = %2.2f\n' % statistics[keys['accuracy']])
        f.write('- AVG LENGTH CORRECT = %2.2f\n' % statistics[keys['avg_length_correct']])
        f.write('- AVG LENGTH WRONG = %2.2f\n' % statistics[keys['avg_length_errors']])
    f.close()

def PrintConfusionMatrix(filename, matrix, classes):
    num_classes = len(classes)
    separator = ';'
    with open(filename, 'w') as f:
        #1st row
        f.write('LABEL/PRED' + separator)
        for i in range(num_classes):
            f.write('%s%s' % (classes[i], separator))
        f.write('\n')
        # Rest of the matrix
        for i in range(num_classes):
            f.write('%s%s' % (classes[i], separator))
            for j in range(num_classes):
                f.write('%d%s' % (matrix[i][j], separator))
            f.write('\n')
    f.close()

data = []
label_set = read_topics_COVID(L6N_folder + L6N_topics_file)
for i in range(5):
#for i in range(1):
    result = predict_COVID(task, model_names[i], dataset_keys[i], label_set)
    data = data + result

matrix, statistics, error_matrix = GetConfusionMatrix(data, label_set)

statistics_filename = os.path.join(PREDICTIONS_DIR, "{}_statistics.txt".format(name))
PrintStatistics(statistics_filename, statistics, label_set)

matrix_filename = os.path.join(PREDICTIONS_DIR, "{}_matrix.csv".format(name))
PrintConfusionMatrix(matrix_filename, matrix, label_set)

matrix_filename = os.path.join(PREDICTIONS_DIR, "{}_matrix_only_errors.csv".format(name))
PrintConfusionMatrix(matrix_filename, error_matrix, label_set)
exit()