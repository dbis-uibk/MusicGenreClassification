#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import log
import argparse
import pickle
import collections
import itertools
import csv

from os.path import splitext, basename
from collections import defaultdict


def read_genre_mappings(mapping_file):
    mapping = defaultdict(dict)
    fieldnames = ['allmusic', 'discogs', 'lastfm', 'tagtraum']
    with open(mapping_file, 'r') as fh:
        reader = csv.DictReader(fh, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
        next(reader)
        for line in reader:
            for dataset, genre in line.items():
                if not genre == '':
                    mapping[dataset][genre] = {k: line[k] for k in line.keys() if line[k] != '' and k != dataset}
    return dict(mapping)


def substitute_genre(n1, n2, val):
    if n1 in genre_map and val in genre_map[n1] and n2 in genre_map[n1][val]:
        return genre_map[n1][val][n2]
    else:
        return ''


parser = argparse.ArgumentParser(description="This script implements task 2 of the MediaEval 2017 challenge.")
parser.add_argument('-c1', '--classifier1', required=True, help='The first main classifier, which is assumed to correspond to the given test dataset.')
parser.add_argument('-c2', '--classifier2', required=True, help='The second main classifier.')
parser.add_argument('-c3', '--classifier3', required=True, help='The third main classifier.')
parser.add_argument('-c4', '--classifier4', required=True, help='The fourth main classifier.')
parser.add_argument('-n1', '--name1', required=True, help='The name of the first dataset.')
parser.add_argument('-n2', '--name2', required=True, help='The name of the second dataset.')
parser.add_argument('-n3', '--name3', required=True, help='The name of the third dataset.')
parser.add_argument('-n4', '--name4', required=True, help='The name of the fourth dataset.')
parser.add_argument('-m', '--mapping', required=True, help='The genre mapping file.')
parser.add_argument('-test', '--test_file', required=True, help='The pickled test file for the relevant dataset.')
parser.add_argument('-o', '--output_file', required=True, help='The predicted classes will be written into this file, which then should be able to be evaluated with the R script provided by the challenge.')
parser.add_argument('-j', '--jobs', default=4, help='Number of parallel Jobs')

args = parser.parse_args()
run_name = splitext(basename(args.test_file))[0]
logger = log.get_logger(run_name)
logger.info('started a run with args: ' + str(args))

logger.debug('reading test file: %s' % args.test_file)
test = pickle.load(open(args.test_file, 'rb'))

print('reading genre mapping: %s' % args.mapping)
genre_map = read_genre_mappings(args.mapping)

logger.debug('reading classifiers')
classifiers = []
logger.debug('    %s' % args.classifier1)
classifiers.append(pickle.load(open(args.classifier1, 'rb')))
logger.debug('    %s' % args.classifier2)
classifiers.append(pickle.load(open(args.classifier2, 'rb')))
logger.debug('    %s' % args.classifier3)
classifiers.append(pickle.load(open(args.classifier3, 'rb')))
logger.debug('    %s' % args.classifier4)
classifiers.append(pickle.load(open(args.classifier4, 'rb')))

logger.debug('scaling test data')
scaler = classifiers[0]['scaler']
xtest  = test.loc[:, 'lowlevel_average_loudness':]
xtest  = scaler.transform(xtest)

logger.debug('predicting main genres via majority vote')
logger.debug('    ... predicting')
# Predict main genre with every classifier
main_predictions = []
for classifier in classifiers:
    clf = classifier['main_classifier']
    mlb = classifier['mlb']
    pred = clf.predict(xtest)
    pred = mlb.inverse_transform(pred)
    main_predictions.append(pred)

# Do voting to get the final result
logger.debug('    ... voting')
main_prediction = []
for p in zip(main_predictions[0], main_predictions[1], main_predictions[2], main_predictions[3]):
    # Map genres
    pred = []
    pred.append(p[0])
    pred.append(tuple(map(lambda x: substitute_genre(args.name2, args.name1, x), p[1])))
    pred.append(tuple(map(lambda x: substitute_genre(args.name3, args.name1, x), p[2])))
    pred.append(tuple(map(lambda x: substitute_genre(args.name4, args.name1, x), p[3])))

    # Vote for the result
    # We are doing multi-label prediction here, so normal majority vote does not work
    # Current solution: Take every main genre that was predicted by at least 50% of classifiers
    res = ()
    predictions = [x for x in pred if not all([y == '' for y in x])]
    n_clf = len(predictions) + 1

    for genre in itertools.chain.from_iterable(predictions):
        if genre in res or genre == '':
            continue

        votes = len([x for x in predictions if genre in x])
        if genre in pred[0]:
            votes += 1

        if votes > n_clf * 0.5:
            res += (genre,)

    # Fallback: If no majority was found, use whatever the classifier for the current data set predicted
    if not res:
        res = p[0]

    main_prediction.append(res)

logger.debug('filling empty main genres with most common')
default_genre = max(classifiers[0]['genre_freq'].items(), key=lambda x: x[1])[0]
empty_indices = [i for i, e in enumerate(main_prediction) if not e]
logger.info('%d/%d (%5.2f%%) samples were not classified and are filled with "%s"' % (len(empty_indices), len(main_prediction), 100*len(empty_indices)/len(main_prediction), default_genre))
for index in empty_indices:
    main_prediction[index] = (default_genre,)

logger.debug('grouping by genre')
genre_dict = collections.defaultdict(list)
for i, (x, ys) in enumerate(zip(xtest, main_prediction)):
    # Run classifier for every identified main genre.
    for main_genre in ys:
        if main_genre in classifiers[0]['empty_genres']:
            continue
        genre_dict[main_genre].append(i)

logger.debug('predicting sub genres')
for genre, indices in genre_dict.items():
    if genre in classifiers[0]['empty_genres']:
        logger.debug('skipping empty genre: %s' % genre)
        continue
    if genre in classifiers[0]['predictive_subgenres']:
        logger.debug('assigning only subgenre(s) to main genre: %s' % genre)
        for index in indices:
            for subgenre in classifiers[0]['predictive_subgenres'][genre]:
                if subgenre not in main_prediction[index]:
                    main_prediction[index] += (subgenre,)
        continue

    logger.debug('\t%s (%d samples)' % (genre, len(indices)))
    x = xtest[indices]
    sub_data = classifiers[0]['subgenre_data'][genre]
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
