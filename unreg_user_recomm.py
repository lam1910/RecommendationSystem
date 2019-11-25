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
print("To run the code smoothly please follow this instruction before running anything.")
print("If this is you first time run this, force stop the program and follow the instruction. If not the program "
      "will stop before you get the item with good rating rated by more than 60% of users in each group and print out "
      "an error")
print("Additional steps:")
print(f"\t1. Download postgresql @ https://www.postgresql.org/download/")
print("\t2. Install postgresql and pgadmin4 (included in most postgresql download) or can download "
      "@ https://www.pgadmin.org/download/ (recommended)")
print("\t3. Run rating_query.sql with psql or on pgadmin4")
import gc

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans

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
print("Data Imported!")

userInfor = userDataset.values
itemInfor = itemDataset.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]].values
mapping = mappingDataset.iloc[:, [0, 1, 2]].values


labelEncoder_userInfor = LabelEncoder()
userInfor[:, 2] = labelEncoder_userInfor.fit_transform(userInfor[:, 2])
userInfor[:, 3] = labelEncoder_userInfor.fit_transform(userInfor[:, 3])
userInfor[:, 4] = labelEncoder_userInfor.fit_transform(userInfor[:, 4])
# male = 1; female = 0
oneHotEncoder = OneHotEncoder(categorical_features = [3, 4])
userInfor = oneHotEncoder.fit_transform(userInfor).toarray()
print("Data Encoded. Preprocessing completed")

#using the elbow method to select K
print()
print("_________________________________")
print("Begin using the elbow method to select K")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 500, n_init = 20, random_state = 0)
    kmeans.fit(userInfor)
    wcss.append(kmeans.inertia_)

f = plt.figure(1)
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

print()
print("Resumed")

#K-means
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 500, n_init = 20, random_state = 0)
y_kmeans = kmeans.fit_predict(userInfor)

print()
print("_________________________________")
print("Train completed. Information:")
print("\tNumber of cluster: %s" %kmeans.n_clusters)
print("\tWCSS: %s" %kmeans.inertia_)

g1 = []
g2 = []
g3 = []

for i in range(len(kmeans.labels_)):
    if kmeans.labels_[i] == 0:
        g1.append(i)
    elif kmeans.labels_[i] == 1:
        g2.append(i)
    else:
        g3.append(i)


groupOne = userDataset.iloc[g1, :].reset_index(drop = True)
groupTwo = userDataset.iloc[g2, :].reset_index(drop = True)
groupThree = userDataset.iloc[g3, :].reset_index(drop = True)

g1_id = groupOne.userid.tolist()
g2_id = groupTwo.userid.tolist()
g3_id = groupThree.userid.tolist()

g1_rating = mappingDataset[mappingDataset.userid.isin(g1_id)].reset_index(drop = True)
g2_rating = mappingDataset[mappingDataset.userid.isin(g2_id)].reset_index(drop = True)
g3_rating = mappingDataset[mappingDataset.userid.isin(g3_id)].reset_index(drop = True)

print()
print("_________________________________")
print("Split rating by group of users completed. Beginning to put data to database.")

del g1, g2, g3, g1_id, g2_id, g3_id
gc.collect()

from sqlalchemy import create_engine
import psycopg2
print('Change information regarding the database according to your preference.')
print("Syntax: dialect+driver://username:password@host:port/database")
engine = create_engine('postgresql://postgres:admin@localhost:5432/movielensgroupuser')


# put dataframe to postgresql, if tables are already existed, replace them with new one
g1_rating.to_sql(name = 'group_one', con = engine, schema = 'groupdata', if_exists = 'replace', index = False)
g2_rating.to_sql(name = 'group_two', con = engine, schema = 'groupdata', if_exists = 'replace', index = False)
g3_rating.to_sql(name = 'group_three', con = engine, schema = 'groupdata', if_exists = 'replace', index = False)

# additional steps:
# download postgresql
# install postgresql and pgadmin4 (recommended) g
# run rating_query.sql with psql or on pgadmin4
print()
print("Warning! Additional steps required to continue. If there is a straight line appear, you do not need to worry"
      ". Else follow the instruction")
try:
    g1_rating = pd.read_sql_table(table_name = 'group_one_good_rating', con = engine, schema = 'groupdata')
    g2_rating = pd.read_sql_table(table_name = 'group_two_good_rating', con = engine, schema = 'groupdata')
    g3_rating = pd.read_sql_table(table_name = 'group_three_good_rating', con = engine, schema = 'groupdata')
    print("_________________________________")
except ValueError:
    print('Program stopped because you do not have the database necessary. There is high chance that you have not run '
          'the sql script yet. To do that please follow the intruction below')
    print("Additional steps:")
    print(f"\t1. Download postgresql @ https://www.postgresql.org/download/")
    print("\t2. Install postgresql and pgadmin4 (included in most postgresql download) or can download "
          "@ https://www.pgadmin.org/download/ (recommended)")
    print("\t3. Run rating_query.sql with psql or on pgadmin4")
    print()




