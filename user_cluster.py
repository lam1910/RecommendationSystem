#importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
userDataset = pd.read_csv(filepath_or_buffer = 'ml-100k/u.user', sep = '|', header = None, names = ['userid', 'age'
    , 'gender', 'occupation', 'zip'], dtype = {'userid': str, 'age': float, 'gender': str, 'occupation': str, 'zip': str})
# exercise/recommendationSystem/ml-100k/u.user
itemDataset = pd.read_csv(filepath_or_buffer = 'ml-100k/u.item', sep = '|', header = None, names = ['itemid'
    , 'movietitle', 'releasedate', 'videoreleasedate', 'IMDbURL', "unknown", "Action", "Adventure", "Animation"
    , "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery"
    , "Romance", "Sci-Fi", "Thriller", "War", "Western"], encoding='ISO-8859-1')
# exercise/recommendationSystem/ml-100k/u.item
mappingDataset = pd.read_csv(filepath_or_buffer = 'ml-100k/u.data', sep = '\t', header = None, names = ['userid'
    , 'movieid', 'rating', 'timestamp'])
# exercise/recommendationSystem/ml-100k/u.data

# convert to list-like
userInfor = userDataset.values
itemInfor = itemDataset.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]].values
mapping = mappingDataset.iloc[:, [0, 1, 2]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_userInfor = LabelEncoder()
userInfor[:, 2] = labelEncoder_userInfor.fit_transform(userInfor[:, 2])
userInfor[:, 3] = labelEncoder_userInfor.fit_transform(userInfor[:, 3])
userInfor[:, 4] = labelEncoder_userInfor.fit_transform(userInfor[:, 4])
# male = 1; female = 0
oneHotEncoder = OneHotEncoder(categorical_features = [3, 4])
userInfor = oneHotEncoder.fit_transform(userInfor).toarray()

#using the elbow method to select K
from sklearn.cluster import KMeans
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

#K-means
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 500, n_init = 20, random_state = 0)
y_kmeans = kmeans.fit_predict(userInfor)

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

g1_rating = g1_rating[g1_rating.rating >= 4].reset_index(drop = True)
g2_rating = g2_rating[g2_rating.rating >= 4].reset_index(drop = True)
g3_rating = g3_rating[g3_rating.rating >= 4].reset_index(drop = True)
