# read stats file for each dataset, gather stats about genres contained

import glob
import csv
import os
import collections
import sys
import distance
from collections import defaultdict

# reads stats files
def read_stats_files(stats_dir, stats_files):
  stats = collections.defaultdict(list)
  for curr_file in stats_files:
    dataset = os.path.split(curr_file)[1].replace('acousticbrainz-mediaeval2017-', '').replace('-train.stats.csv', '') 
    with open(curr_file, 'r') as fh:
      fieldnames = ['tag', 'recordings', 'releasegroups']
      reader = csv.DictReader(fh, delimiter='\t', fieldnames=fieldnames)
      next(reader, None)
      for line in reader:
        stats[dataset].append(line) 
  return stats


# extracts main genres for given dataset
def extract_genres(stats, dataset):
  return [x['tag'] for x in stats[dataset] if not '---' in x['tag']]

# extracts subgenres for given dataset
def extract_subgenres(stats, dataset):
  return [x['tag'] for x in stats[dataset] if '---' in x['tag']]

# returns diff of overlap and genre-list
def get_genre_difference(genres, overlap):
  overlap_genres = []
  for curr_overlap in overlap:
    overlap_genres.append(curr_overlap['dataset1'])
    overlap_genres.append(curr_overlap['dataset2'])
  return list(set(set(genres) - set(overlap_genres)))
      

# fuzzy comparison of genres
def get_overlapping_genres(stats, subgenres=False, exclude_main_genres=False, fuzzy=True):
  result = []
  datasets = list(stats.keys())
  for dataset1 in datasets:
    for dataset2 in datasets[datasets.index(dataset1)+1:]:
      if subgenres:
        genres_dataset1 = extract_subgenres(stats, dataset1)
        genres_dataset2 = extract_subgenres(stats, dataset2)
        if exclude_main_genres:
          genres_dataset1 = [x.split('---')[1] for x in genres_dataset1]
          genres_dataset2 = [x.split('---')[1] for x in genres_dataset2]
      else:
        genres_dataset1 = extract_genres(stats, dataset1)
        genres_dataset2 = extract_genres(stats, dataset2)
      
      overlap = []
      for genre1 in genres_dataset1:
        for genre2 in genres_dataset2:
          dist = distance.levenshtein(genre1, genre2)
          if fuzzy:
            if dist >= 0 and dist < 2:
              overlap.append({'dataset1': genre1, 'dataset2': genre2}) 
          else:
            if dist == 0:
              overlap.append({'dataset1': genre1, 'dataset2': genre2}) 

      only_dataset1 = get_genre_difference(genres_dataset1, overlap)
      only_dataset2 = get_genre_difference(genres_dataset2, overlap)
      print("------------------------")
      print(dataset1, dataset2)
      print("overlap: ", overlap)
      print("only in dataset1: ", sorted(only_dataset1))
      print("only in dataset2: ", sorted(only_dataset2))
               
      result.append({'dataset1': dataset1, 'dataset2': dataset2, 'overlap': overlap})
  return result

# prints genres and subgenres as tree
def write_tag_hierarchies(stats, output_file):
  with open(output_file, 'w') as fh:
    for dataset, tags in stats.items():
      fh.write("---------")
      fh.write(dataset)
      fh.write("---------\n\n")
      for curr_tag in tags:
         if not "---" in curr_tag['tag']:
            fh.write(curr_tag['tag'] + "\n")
         else:
            fh.write("|--" + curr_tag['tag'].split('---')[1] + "\n")
      fh.write("\n\n")
 
# extracts mappings from overlaps
def get_genre_mappings(data):
  same_as = defaultdict(dict) 
  for curr_combination in data:
    if 'overlap' in curr_combination:
      for curr_overlap in curr_combination['overlap']:
        same_as[curr_overlap['dataset1']].update({curr_combination['dataset2']: curr_overlap['dataset2']}) 
  return same_as

# creates mapping csv file 
def write_mapping_file(stats, output_file):
  same_as = get_genre_mappings(stats)
  fieldnames = ['allmusic', 'discogs', 'lastfm', 'tagtraum']
  with open(output_file, 'w') as fh:
    writer = csv.DictWriter(fh, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames, restval=None)
    writer.writeheader()
    for curr_genre, mapping in same_as.items():
      if len(mapping) > 1:
        writer.writerow(mapping)


# reads mapping csv in two-way map
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


#########################

stats_dir = '/mnt/eva_nas/mediaeval2017/groundtruth_data/'
stats_files = glob.glob(stats_dir + "*stats.csv")
data = read_stats_files(stats_dir, stats_files)
hierarchies_file = 'tag_hierarchies.txt'
#write_tag_hierarchies(data, hierarchies_file)
#print("tag hierarchies of datasets written to: " + hierarchies_file)
 
genres_overlap = get_overlapping_genres(data, fuzzy=True)
subgenres_overlap = get_overlapping_genres(data, subgenres=True, fuzzy=True)
#subgenres_overlap = []
overlap = genres_overlap + subgenres_overlap
write_mapping_file(overlap, 'genre_mapping.csv')
read_genre_mappings('genre_mapping.csv')
