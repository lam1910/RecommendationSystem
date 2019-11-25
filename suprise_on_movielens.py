# This file is for a single customer recommendation based on the movie just rated highly
# similar to if you like that, you will love this

import io  # needed because of weird encoding of u.item file
import random as rd # needed for utility function, has no real value, if cannot run program can just take one uid
                    # from u.user and delete this

from surprise import Dataset
from surprise import KNNBaseline
from surprise import get_dataset_dir


global genre
genre = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama"
        , "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

"""
u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of 
	         user id | item id | rating | timestamp. 
              The time stamps are unix seconds since 1/1/1970 UTC   

u.info     -- The number of users, items, and ratings in the u data set.

u.item     -- Information about the items (movies); this is a tab separated
              list of
              movie id | movie title | release date | video release date |
              IMDb URL | unknown | Action | Adventure | Animation |
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western |
              The last 19 fields are the genres, a 1 indicates the movie
              is of that genre, a 0 indicates it is not; movies can be in
              several genres at once.
              The movie ids are the ones used in the u.data data set.

u.genre    -- A list of the genres.

u.user     -- Demographic information about the users; this is a tab
              separated list of
              user id | age | gender | occupation | zip code
              The user ids are the ones used in the u.data data set.

u1.base    -- The data sets u1.base and u1.test through u5.base and u5.test
u1.test       are 80%/20% splits of the u data into training and test data.
u2.base       Each of u1, ..., u5 have disjoint test sets; this if for
u2.test       5 fold cross validation (where you repeat your experiment
u3.base       with each training and test set and average the results).
u3.test       These data sets can be generated from u.data by mku.sh.
u4.base
u4.test
u5.base
u5.test
"""


def read_item_names():
    # read the u.item file from MovieLens 100-k dataset and
    # return two mappings to convert raw ids into movie names and movie names into raw ids.
    file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.item'
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid


def get_item_good_review(uid):
    # get the item with the latest timestamp that the user with uid rate 5 star.
    # return said itemid (iid) if found
    # return None if not found
    true_uid = str(uid)
    file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.data'
    iugrating = []
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('\t')
            if line[0] == true_uid and int(line[2]) == 5:
                iugrating.append(line[1])

    try:
        return iugrating[len(iugrating) - 1]
    except IndexError:
        return None


def get_random_user():
    # utility function, has no real value
    # return a single, random uid
    file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.user'
    uids = []
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            uids.append(line[0])

    try:
        return rd.choice(uids)
    except IndexError:
        return None


def get_movie_infor(iid):
    # utility function, to be used on movie_to_string()
    # return the line on u.item that has movieid = iid
    # return None if not found
    true_iid = str(iid)
    file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.item'
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            if line.split('|')[0] == true_iid:
                return line

    return None


def get_genre_matrix(iid):
    # utility function, to be used on movie_to_string()
    # return the representation of genre as a matrix of '0' and '1'
    true_iid = str(iid)
    tmp = get_movie_infor(true_iid).split('|')[5:]
    if tmp[-1] == '0\n':
        tmp[-1] = '0'
    else:
        tmp[-1] = '1'
    return tmp

def movie_to_string(iid):
    # utility function, to print the true information of the movie not just the matrix of genre
    # print directly no return
    true_iid = str(iid)
    movie_infor = get_movie_infor(true_iid).split('|')
    mask = get_genre_matrix(true_iid)
    line = movie_infor[0] + ' - ' + movie_infor[1] + ' - ' + movie_infor[2] + ' - ' + movie_infor[3] \
           + ' - ' + movie_infor[4] + ' - '

    for i in range(len(genre)):
        if mask[i] == '1':
            line += genre[i]
            line += ', '

    line += '.\n'
    print(line)


# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')

trainset = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options = sim_options)
algo.fit(trainset)


# Read the mappings raw id <-> movie name
rid_to_name, name_to_rid = read_item_names()

#the user id 888 is the example

random_uid = get_random_user()
# result of get_random_user():
# 499 (not through whole run), 914 (not through whole run), 888, 346 (not through whole run), ...
raw_iid = get_item_good_review(random_uid)  # 237
item_name = rid_to_name[raw_iid]            # Jerry Macguire (1996)
inner_id = algo.trainset.to_inner_iid(raw_iid)

# Retrieve inner ids of the nearest neighbors of the item.
item_neighbors = algo.get_neighbors(inner_id, k = 5)

item_neighbors = (algo.trainset.to_raw_iid(i_inner_id) for i_inner_id in item_neighbors)
item_neighbors = list(item_neighbors)
# = [255, 215, 204, 22, 230]
# or [My Best Friend's Wedding (1997), Field of Dreams (1989), Back to the Future (1985), Braveheart (1995)
# , Star Trek IV: The Voyage Home (1986)]


print()
print('___________________')
print('User ID: %s' %random_uid)
print('Last 5-star review: %(i)s - %(n)s' %{'i': raw_iid, 'n': item_name})
print('Movie %s detail: ' %item_name)
movie_to_string(raw_iid)
print('Similar film that you could like: ')

for movie in item_neighbors:
    movie_to_string(movie)

