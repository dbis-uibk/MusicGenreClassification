#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import log
import argparse
import pickle
import collections
import numpy as np

from os.path import splitext, basename

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

from etl import TargetExtractor
from classifiers import get_classifier_by_name

parser = argparse.ArgumentParser(description="This script implements task 1 of the MediaEval 2017 challenge.")
parser.add_argument('-i', '--input_file', required=True)
parser.add_argument('-mc', '--maingenre_classifier', default='linearsvc', help='The classifier to be used in the main genre. one of "rf", "mlp", "xgboost". Defaults to "mlp".')
parser.add_argument('-sc', '--subgenre_classifier', default='et', help='The classifier to be used in the sub genre. one of "rf", "mlp", "xgboost". Defaults to "mlp".')
parser.add_argument('-test', '--test_file', help='The pickled test file for the relevant dataset. If not provided, this script will use the train_test_split function of scikit.')
parser.add_argument('-m', '--model_file', help='The file the trained model should be written to.')
parser.add_argument('-o', '--output_file', required=True, help='The predicted classes will be written into this file, which then should be able to be evaluated with the R script provided by the challenge.')
parser.add_argument('-j', '--jobs', default=4, help='Number of parallel Jobs')

args = parser.parse_args()
run_name = '%s_%s' % (splitext(basename(args.input_file))[0], args.maingenre_classifier)
logger = log.get_logger(run_name)
logger.info('started a run with args: ' + str(args))

logger.debug('reading input file: %s' % args.input_file)
df = pickle.load(open(args.input_file, 'rb'))
if args.test_file:
    logger.debug('using test file: %s' % args.test_file)
    train = df
    test = pickle.load(open(args.test_file, 'rb'))
else:
    logger.debug('splitting training file for testing')
    train, test = train_test_split(df)

mlb = MultiLabelBinarizer()

all_columns = list(train)
ignored_columns = train[[x for x in all_columns if '__' in x]]
remaining_columns = [x for x in all_columns if x not in ignored_columns]
xtrain = train[remaining_columns]
xtest  =  test[remaining_columns]

logger.debug('extracting main genres')
target_extractor_train = TargetExtractor(train)
ytrain = target_extractor_train.get_labels_for_main(mlb)
target_extractor_test = TargetExtractor(test)
ytest = target_extractor_test.get_labels_for_main(mlb)

logger.debug('scaling main features')
scaler = MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest  = scaler.transform(xtest)

logger.debug('fitting main genre with classifier: %s' % args.maingenre_classifier)
main_classifier = get_classifier_by_name(args.maingenre_classifier, n_jobs=int(args.jobs))
main_classifier.fit(xtrain, ytrain)

logger.debug('predicting main genres')
main_prediction = main_classifier.predict(xtest)
main_prediction = mlb.inverse_transform(main_prediction)

def are_samples_identical(data):
    iterator = iter(data)
    try:
        first = str(next(iterator))
    except StopIteration:
        return True
    return all(first == str(rest) for rest in iterator)

logger.debug('fitting sub genres with classifier: %s' % args.subgenre_classifier)
subgenre_data = {}
genre_freq = {}
empty_genres = set()
predictive_subgenres = {}
for i, genre in enumerate(mlb.classes_):
    subgenre_data[genre] = {}
    logger.debug('\t%s' % genre)
    smlb = MultiLabelBinarizer()
    xtrain_s, ytrain_s = target_extractor_train.get_labels_for_sub(genre, smlb)
    xtrain_s = xtrain_s[remaining_columns]
    if ytrain_s[0].size == 0:
        logger.debug('\t\tno subgenres or identical subgenres found for main genre %s' % genre)
        empty_genres.add(genre)
        continue
    if are_samples_identical(ytrain_s):
        logger.debug('\t\tonly identical subgenres found for main genre %s' % genre)
        predictive_subgenres[genre] = smlb.inverse_transform(np.array([ytrain_s[0]]))[0]
        continue
    xtrain_s = scaler.transform(xtrain_s)
    genre_freq[genre] = len(xtrain_s)
    sclf = get_classifier_by_name(args.subgenre_classifier, n_jobs=int(args.jobs))
    sclf.fit(xtrain_s, ytrain_s)
    subgenre_data[genre]['classifier'] = sclf
    subgenre_data[genre]['mlb'] = smlb

if args.model_file:
    logger.debug('saving trained model to %s' % args.model_file)
    data = {
        'mlb': mlb,
        'scaler': scaler,
        'predictive_subgenres': predictive_subgenres,
        'main_classifier': main_classifier,
        'subgenre_data': subgenre_data,
        'empty_genres': empty_genres,
        'genre_freq': genre_freq
    }
    pickle.dump(data, open(args.model_file,'wb'))
    exit()

logger.debug('predictive subgenres:')
[logger.debug('  %s: %s' % x) for x in predictive_subgenres.items()]

logger.debug('filling empty main gernes with most common')
default_genre = max(genre_freq.items(), key=lambda x: x[1])[0]
empty_indices = [i for i, e in enumerate(main_prediction) if not e]
logger.info('%d/%d (%5.2f%%) samples were not classified and are filled with "%s"' % (len(empty_indices), len(main_prediction), 100*len(empty_indices)/len(main_prediction), default_genre))
for index in empty_indices:
    main_prediction[index] = (default_genre,)

logger.debug('grouping by genre')
genre_dict = collections.defaultdict(list)
for i, (x, ys) in enumerate(zip(xtest, main_prediction)):
    # Run classifier for every identified main genre.
    for main_genre in ys:
        if main_genre in empty_genres:
            continue
        genre_dict[main_genre].append(i)
       
logger.debug('predicting sub genres')
for genre, indices in genre_dict.items():
    if genre in empty_genres:
        logger.debug('skipping empty genre: %s' % genre)
        continue
    if genre in predictive_subgenres:
        logger.debug('assigning only subgenre(s) to main genre: %s' % genre)
        for index in indices:
            for subgenre in predictive_subgenres[genre]:
                if subgenre not in main_prediction[index]:
                    main_prediction[index] += (subgenre,)
        continue

    logger.debug('\t%s (%d samples)' % (genre, len(indices)))
    x = xtest[indices]
    sub_data = subgenre_data[genre]
    y = sub_data['classifier'].predict(x)
    y = sub_data['mlb'].inverse_transform(y)

    for index, subgenre_predictions in zip(indices, y):
        for subgenre in subgenre_predictions:
            if subgenre not in  main_prediction[index]:
                main_prediction[index] += (subgenre,)

logger.info('writing results to %s' % args.output_file)
y_ids = test.loc[:, '__recordingmbid']
with open(args.output_file, 'w') as out:
    for rec_id, predictions in zip(y_ids, main_prediction):
        out.write('%s\t%s\n' % (rec_id, '\t'.join(predictions)))

if not args.test_file:
    logger.info('writing ground truth to %s' % args.output_file + '_gt')
    with open(args.output_file + '_gt', 'w') as gt_out:
        gt_out.write('dummy\n')
        gt_genres = target_extractor_test.get_labels_all()
        for rec_id, g in zip(y_ids, gt_genres):
            gt_out.write('%s\tdummy\t%s\n' % (rec_id, '\t'.join(g)))
