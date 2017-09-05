# MusicGenreClassification
This repository contains the implementation of our (team DBIS) solution to the AcousticBrainz Genre Task: Content-based music genre recognition from multiple sources as part of MediaEval 2017.

# Repository Structure
Our solutions to subtasks 1 and 2 can be found in the `task1` and `task2` folders, respectively. The main code files are `task1.py` and `task2.py`.

The detailed list of features we used to train our classifiers is given in the file `features.txt`.

# Datasets and Classifiers
Pickled datasets for subtask 1 and pre-trained classifiers for subtask 2 can be downloaded at the following locations:

* Datasets for subtask 1: https://dbis-owncloud.uibk.ac.at/index.php/s/PL7iQzaZBGpzWhm/download?path=%2F&files=task1_datasets.zip
* Classifiers for subtask 2: https://dbis-owncloud.uibk.ac.at/index.php/s/PL7iQzaZBGpzWhm/download?path=%2F&files=task2_classifiers.zip

# Examples
Example usage for `task1.py`:

```
./task1.py -i discogs.pickle -o out.txt
```

Example usage for `task2.py`:

```
./task2.py -c1 classifiers/discogs.pickle -c2 classifiers/allmusic.pickle -c3 classifiers/lastfm.pickle -c4 classifiers/tagtraum.pickle -n1 discogs -n2 allmusic -n3 lastfm -n4 tagtraum -m genre_mapping.csv -test data/discogs.pickle -o out.txt
```
