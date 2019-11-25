"""
import numpy as np
from surprise import KNNBasic
from surprise import Dataset

# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# Retrieve the trainset.
trainset = data.build_full_trainset()

# Build an algorithm, and train it.
algo = KNNBasic()
algo.fit(trainset)

# we can now query for specific predicions
uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
iid = str(302)  # raw item id (as in the ratings file). They are **strings**!

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=4, verbose=True)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import time
from lightfm.datasets import fetch_movielens

movielens = fetch_movielens()

train = movielens['train']
test = movielens['test']

from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

model = LightFM(learning_rate=0.05, loss='warp', no_components=64, item_alpha=0.001)

model.fit_partial(train, item_features=movielens['item_features'], epochs=20 )

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()

train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
_, item_embeddings = model.get_item_representations(movielens['item_features'])


from annoy import AnnoyIndex

factors = item_embeddings.shape[1] # Length of item vector that will be indexed
annoy_idx = AnnoyIndex(factors)
for i in range(item_embeddings.shape[0]):
    v = item_embeddings[i]
    annoy_idx.add_item(i, v)

annoy_idx.build(10) # 10 trees
annoy_idx.save('movielens_item_Annoy_idx.ann')

def nearest_movies_annoy(movie_id, index, n=10, print_output=True):
    nn = index.get_nns_by_item(movie_id, 10)
    if print_output:
        print('Closest to %s : \n' % movielens['item_labels'][movie_id])
    titles = [movielens['item_labels'][i] for i in nn]
    if print_output:
        print("\n".join(titles))

nearest_movies_annoy(0, annoy_idx)

import nmslib

# initialize a new nmslib index, using a HNSW index on Cosine Similarity
nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
nms_idx.addDataPointBatch(item_embeddings)
nms_idx.createIndex(print_progress=True)

def nearest_movies_nms(movie_id, index, n=10, print_output=True):
    nn = index.knnQuery(item_embeddings[movie_id], k=10)
    if print_output == True:
        print('Closest to %s : \n' % movielens['item_labels'][movie_id])
    titles = [movielens['item_labels'][i] for i in nn[0]]
    if print_output == True:
        print("\n".join(titles))

nearest_movies_nms(90, nms_idx, n=10)
nearest_movies_nms(90, nms_idx, n=10, print_output=False)

norms = np.linalg.norm(item_embeddings, axis=1)
max_norm = norms.max()
extra_dimension = np.sqrt(max_norm ** 2 - norms ** 2)
norm_data = np.append(item_embeddings, extra_dimension.reshape(norms.shape[0], 1), axis=1)

# First an Annoy index:

user_factors = norm_data.shape[1]
annoy_member_idx = AnnoyIndex(user_factors)  # Length of item vector that will be indexed

for i in range(norm_data.shape[0]):
    v = norm_data[i]
    annoy_member_idx.add_item(i, v)

annoy_member_idx.build(10)
# Now an NMS index

nms_member_idx = nmslib.init(method='hnsw', space='cosinesimil')
nms_member_idx.addDataPointBatch(norm_data)
nms_member_idx.createIndex(print_progress=True)

# Define our user vectors

_, user_embeddings = model.get_user_representations()

def sample_recommendation(user_ids, model, data, n_items=10, print_output=True):
    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        top_items = [data['item_labels'][i] for i in annoy_member_idx.get_nns_by_vector(np.append(user_embeddings[user_id], 0), 50)]
        if print_output == True:
            print("User %s" % user_id)
            print("     Known positives:")
            try:
                for x in known_positives[:5]:
                    print("        %s" % x)
            except IndexError:
                for x in known_positives[:]:
                    print("        %s" % x)

            print("     Recommended:")

            try:
                for x in known_positives[:5]:
                    print("        %s" % x)
            except IndexError:
                for x in known_positives[:]:
                    print("        %s" % x)

sample_recommendation([3,25,450], model, movielens, print_output=True)

print("Compare")

def sample_recommendation_original(model, data, user_ids, print_output=True):

    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]
        if print_output == True:
            print("User %s" % user_id)
            print("     Known positives:")

            try:
                for x in known_positives[:5]:
                    print("        %s" % x)
            except IndexError:
                for x in known_positives[:]:
                    print("        %s" % x)

            print("     Recommended:")

            try:
                for x in known_positives[:5]:
                    print("        %s" % x)
            except IndexError:
                for x in known_positives[:]:
                    print("        %s" % x)

sample_recommendation_original(model, movielens, [3, 25, 450])