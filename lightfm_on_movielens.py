# read data
"""
u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of
	         user id | item id | rating | timestamp.
              The time stamps are unix seconds since 1/1/1970 UTC
              sep = '\t'

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
              sep = '|'
              video release date of all item == nan => removed

u.genre    -- A list of the genres.

u.user     -- Demographic information about the users; this is a tab
              separated list of
              user id | age | gender | occupation | zip code
              The user ids are the ones used in the u.data data set.
              sep = '|'

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
import pandas as pd
import random as rd

from sklearn.model_selection import train_test_split

userDataset = pd.read_csv(filepath_or_buffer = 'ml-100k/u.user', sep = '|', header = None, names = ['userid', 'age'
    , 'gender', 'occupation', 'zip'])
# exercise/recommendationSystem/ml-100k/u.user
itemDataset = pd.read_csv(filepath_or_buffer = 'ml-100k/u.item', sep = '|', header = None, names = ['itemid'
    , 'movietitle', 'releasedate', 'videoreleasedate', 'IMDbURL', "unknown", "Action", "Adventure", "Animation"
    , "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery"
    , "Romance", "Sci-Fi", "Thriller", "War", "Western"], encoding='ISO-8859-1')
# exercise/recommendationSystem/ml-100k/u.item
mappingDataset = pd.read_csv(filepath_or_buffer = 'ml-100k/u.data', sep = '\t', header = None, names = ['userid'
    , 'movieid', 'rating', 'timestamp'])
# exercise/recommendationSystem/ml-100k/u.data


def getRandomUser():
    userList = userDataset.userid.tolist()
    return rd.choice(userList)

def getRandomItem():
    itemList = itemDataset.itemid.tolist()
    return rd.choice(itemList)
# convert to list-like
userInfor = userDataset.values
itemInfor = itemDataset.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]].values
mapping = mappingDataset.iloc[:, [0, 1, 2]].values
mapping_train, mapping_test = train_test_split(mapping, test_size = 0.2, random_state = 0)



# build lightfm dataset
from lightfm import data
dataset = data.Dataset()

dataset.fit(users = (userInfor[:, 0]), items = (itemInfor[:, 0]), user_features = (userInfor[:, 1])
            , item_features = (itemInfor[:, 1]))
dataset.fit_partial(user_features = userInfor[:, 2], item_features = itemInfor[:, 2])
dataset.fit_partial(user_features = userInfor[:, 3], item_features = itemInfor[:, 3])
dataset.fit_partial(user_features = userInfor[:, 4], item_features = itemInfor[:, 4])
dataset.fit_partial(item_features = itemInfor[:, 6])
dataset.fit_partial(item_features = itemInfor[:, 7])
dataset.fit_partial(item_features = itemInfor[:, 8])
dataset.fit_partial(item_features = itemInfor[:, 9])
dataset.fit_partial(item_features = itemInfor[:, 10])
dataset.fit_partial(item_features = itemInfor[:, 11])
dataset.fit_partial(item_features = itemInfor[:, 12])
dataset.fit_partial(item_features = itemInfor[:, 13])
dataset.fit_partial(item_features = itemInfor[:, 14])
dataset.fit_partial(item_features = itemInfor[:, 15])
dataset.fit_partial(item_features = itemInfor[:, 16])
dataset.fit_partial(item_features = itemInfor[:, 17])
dataset.fit_partial(item_features = itemInfor[:, 18])
dataset.fit_partial(item_features = itemInfor[:, 19])
dataset.fit_partial(item_features = itemInfor[:, 20])
dataset.fit_partial(item_features = itemInfor[:, 21])


# build feature
itemFeatures = dataset.build_item_features(((item[0], [item[1], item[2], item[3], item[4], item[8], item[6], item[7]
    , item[8], item[9], item[10], item[11], item[12], item[13], item[14], item[15], item[16], item[17]
    , item[18], item[19], item[20], item[21]]) for item in itemInfor), normalize = False)
userFeatures = dataset.build_user_features(((user[0], [user[1], user[2], user[3], user[4]]) for user in userInfor)
                                           , normalize = False)
# build interaction
mappingFeatures = dataset.build_interactions(((mappingi[0], mappingi[1], mappingi[2]) for mappingi in mapping))
mappingFeatures_train = dataset.build_interactions(((mappingi[0], mappingi[1], mappingi[2])
                                                    for mappingi in mapping_train))
mappingFeatures_test = dataset.build_interactions(((mappingi[0], mappingi[1], mappingi[2])
                                                   for mappingi in mapping_test))

from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

model = LightFM(learning_schedule = 'adagrad', loss='bpr')
model.fit(mappingFeatures_train[0], item_features = itemFeatures, user_features = userFeatures
          , sample_weight = mappingFeatures_train[1], epochs = 30, num_threads = 2, verbose = True)


print("Train precision at 5th: %.2f"
      % precision_at_k(model, mappingFeatures_train[0], item_features = itemFeatures
                       , user_features = userFeatures, k = 5).mean())
print("Test precision at 5th: %.2f"
      % precision_at_k(model, mappingFeatures_test[0], item_features = itemFeatures
                       , user_features = userFeatures, k = 5).mean())

print("ROC AUC metric for train: %.2f"
      % auc_score(model, mappingFeatures_train[0], item_features = itemFeatures
                       , user_features = userFeatures).mean())

print("ROC AUC metric for test: %.2f"
      % auc_score(model, mappingFeatures_test[0], item_features = itemFeatures
                       , user_features = userFeatures).mean())
