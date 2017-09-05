# -*- coding: utf-8 -*-

import collections

class TargetExtractor:
    """
    Extracts target tuples for multi-label classification from the given dataframe.
    Filters the given dataframe on a main genre for subgenre classification.
    :param df: The source dataframe. Genre labels are expected to be located in columns 2 to 2 + num_genres.
    :param df: The number of genre columns in the dataframe.
    """
    def __init__(self, df):
        self.df = df
        self.genre_columns = [x for x in list(df) if '__genre' in x]
        self.subgenre_dict = collections.defaultdict(list)

    def get_labels_for_main(self, mlb):
        """
        Creates target vectors for main genres for the given dataframe.
        :return: A tuple of (dataframe, target vectors)
        """
        targetlist = []
        for index, row in enumerate(self.df[self.genre_columns].itertuples()):
            curtarget = ()
            for y in row:
                if y and type(y) == str:
                    if y.find('---') != -1:
                        y = y[:y.find('---')]

                    if not y in curtarget:
                        curtarget = curtarget + (y,)
                    self.subgenre_dict[y].append(index)

            targetlist.append(curtarget)

        if not hasattr(mlb, 'classes_'):
            mlb.fit(targetlist)

        target = mlb.transform(targetlist)
        return target

    def get_labels_for_sub(self, filter, mlb):
        """
        Creates target vectors for subgenres for the given dataframe and main genre.
        :param filter: The main genre to filter on.
        :return: A tuple of (filtered dataframe, target vectors)
        """

        indices = self.subgenre_dict[filter]
        dframe = self.df.iloc[indices]

        targetlist = []

        for row in dframe[self.genre_columns].itertuples():
            curtarget = tuple()
            for y in row:
                if y and type(y) == str and y.startswith(filter):
                    if '---' not in y:
                        continue
                    if not y in curtarget:
                        curtarget += (y,)
            if not curtarget: 
                curtarget = (filter,)
            targetlist.append(curtarget)

        target = mlb.fit_transform(targetlist)
        return dframe, target

    def get_labels_all(self):
        target = []

        for row in self.df[self.genre_columns].itertuples():
            curtarget = ()
            for y in row:
                if y and type(y) == str:
                    if not y in curtarget:
                        curtarget = curtarget + (y,)

            target.append(curtarget)

        return target
