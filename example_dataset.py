
import os
import zipfile
import csv

import requests

import json
from itertools import islice

from lightfm.data import Dataset
from lightfm import LightFM

import pandas as pd



def _download(url: str, dest_path: str):

    req = requests.get(url, stream=True)
    req.raise_for_status()

    with open(dest_path, "wb") as fd:
        for chunk in req.iter_content(chunk_size=2 ** 20):
            fd.write(chunk)


def get_data():

    ratings_url = ("http://www2.informatik.uni-freiburg.de/" "~cziegler/BX/BX-CSV-Dump.zip")

    if not os.path.exists("data"):
        os.makedirs("data")

        _download(ratings_url, "data/data.zip")

    with zipfile.ZipFile("data/data.zip") as archive:

            ratingsRD = csv.DictReader(
                (x.decode("utf-8", "ignore") for x in archive.open("BX-Book-Ratings.csv")),
                delimiter=";")
            booksRD = csv.DictReader(
                (x.decode("utf-8", "ignore") for x in archive.open("BX-Books.csv")), delimiter=";")
            book_ratings = list(ratingsRD)
            books = list(booksRD)
            return  book_ratings, books



def get_ratings():
    return get_data()[0]


def get_book_features():
    return get_data()[1]


ratings, book_features = get_data()

for line in ratings:
    print(json.dumps(line, indent=4))

df_ratings = pd.DataFrame(columns = ['UserId', 'ISBN', 'Rating'])

for line in islice(book_features, 1):
    print(json.dumps(line, indent=4))


dataset = Dataset()
dataset.fit((x['User-ID'] for x in get_ratings()),
            (x['ISBN'] for x in get_ratings()))

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

dataset.fit_partial(items=(x['ISBN'] for x in get_book_features()),
                    item_features=(x['Book-Author'] for x in get_book_features()))

(interactions, weights) = dataset.build_interactions(((x['User-ID'], x['ISBN'])
                                                      for x in get_ratings()))

print(repr(interactions))

item_features = dataset.build_item_features(((x['ISBN'], [x['Book-Author']])
                                              for x in get_book_features()))
print(repr(item_features))

model = LightFM(loss='bpr')
model.fit(interactions, item_features=item_features)