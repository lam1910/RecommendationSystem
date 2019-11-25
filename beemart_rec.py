import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lightfm import LightFM 
from lightfm.data import Dataset
from lightfm.evaluation import auc_score
from lightfm.evaluation import recall_at_k 
from lightfm.evaluation import precision_at_k
from sklearn.model_selection import train_test_split

import scipy.sparse
from scipy import spatial
    
og_table = pd.read_excel('20191113_Khachhang_Product_Beemart/order_khachang_sp.xlsx')
# exercise/recommendationSystem/20191113_Khachhang_Product_Beemart/order_khachang_sp.xlsx

action_count = og_table.groupby(['CustomerId']).agg({'ProductId': 'count'})
action_count.reset_index(inplace = True)
action_count = action_count[action_count['ProductId'] >=10]

strimmed_table = og_table[og_table['CustomerId'].isin(action_count['CustomerId'])]

items = strimmed_table[['ProductId',    'Productname']]
items = items.drop_duplicates()
items.reset_index(drop = True, inplace = True)

users = strimmed_table[['CustomerId',   'CustomerName']]
users = users.drop_duplicates()
users.reset_index(drop = True, inplace = True)

ratings = strimmed_table[['CustomerId', 'ProductId']]
ratings = ratings.drop_duplicates()
ratings.reset_index(drop = True, inplace = True)

ratings_train, ratings_test = train_test_split(ratings, test_size=0.1)

print(users.shape)
users.head()

print(ratings.shape)
ratings.head()

print(items.shape)
items.head()


"""
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp'];;;[]
ratings_train = pd.read_csv('./ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('./ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
"""


#ratings_train[["user_id","movie_id"]] = ratings_train[["user_id","movie_id"]].astype(str)
#ratings_test[["user_id","movie_id"]] = ratings_test[["user_id","movie_id"]].astype(str)

#ratings_train.shape, ratings_test.shape

#number of unique user and item
n_users = users.shape[0]
n_items = items.shape[0]


#not a good move?
#data_matrix = np.zeros((n_users, n_items))

model = LightFM(loss = 'bpr')

dataset_new = Dataset()
X = set(users.to_numpy().flatten())
#print (X)
Y = set(items.to_numpy().flatten())
dataset_new.fit(users = (x for x in users["CustomerId"]),
                items = (x for x in items["ProductId"]))



(interactions_matrix_train, interaction_weights_train) = \
    dataset_new.build_interactions(((row[1]['CustomerId'], row[1]['ProductId']) for row in ratings_train.iterrows()))

(interactions_matrix_test, interaction_weights_test) = \
    dataset_new.build_interactions(((row[1]['CustomerId'], row[1]['ProductId']) for row in ratings_test.iterrows()))


model.fit(interactions_matrix_train, epochs = 30, num_threads = 2, verbose = True )

#print("SCORE:",auc_score(model, interactions_matrix_test, num_threads=10).mean())

auc = auc_score(model, interactions_matrix_test, interactions_matrix_train, preserve_rows=False, num_threads=1
                , check_intersections=True).mean()

print("AUC SCORE: ", auc)


"""
recall_at_k(model, interactions_matrix_test, train_interactions=interactions_matrix_train, k=10, user_features=None
            , item_features=None, preserve_rows=False, num_threads=1, check_intersections=True).mean()
"""
user_precision = precision_at_k(model, interactions_matrix_test, train_interactions=interactions_matrix_train, k=9
                                , user_features=None, item_features=None, preserve_rows=False, num_threads=1
                                , check_intersections=True)
