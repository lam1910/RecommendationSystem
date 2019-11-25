def fcp(object, mapping, item_features = None, user_features = None):
    # pre-condition: object had been fitted with dataset used to constructed mapping, item_features and user_features.
    # parameters:
    #   object = lightfm model had been fitted with dataset. used to extract ranking of item of user userid (y_pred)
    #   mapping = mapping matrix. used to extract the rating of item of user userid (y_true)
    #   constructed by
    #   mapping = dataset.build_interactions(((mappingi[0], mappingi[1], mappingi[2])
    #                       for mappingi in mapping))[1]
    #       with mapping is the nparray from dataset from mapping file (userid, itemid, rating)
    #   item_features, user_features None or constructed by build_item_features or build_user_features of the same datset.
    # return: fcp of type float = n_c / n_all with:
    #               n_c: number of correct pair.
    #               n_all: number of all pair
    #         if error occurred, return -1
    # has not test all situations. only test for model not fitted any dataset, mapping did not define, mapping does not
    # construct from dataset.build_interactions(((mappingi[0], mappingi[1], mappingi[2]) for mappingi in mapping))[1]
    # did not test for model and mapping fir different dataset, but since mapping does not construct from
    # dataset.build_interactions(((mappingi[0], mappingi[1], mappingi[2]) for mappingi in mapping))[1] produce Exception
    # , not specific error, i just catch Exception.
    try:
        y_pred = object.predict_rank(test_interactions = mapping[0], item_features = item_features
                                , user_features = user_features).toarray()
        y_true = mapping[1].toarray()

        n_user = y_true.shape[0]
        n_item = y_true.shape[1]

        n_correct = 0
        n_all = 0
        for userid in range(n_user):
            tmp = y_pred[userid, :]
            temp = y_true[userid, :]
            n_actual_record = []                # list of itemid that user userid actually rated
            for itemid in range(n_item):
                if temp[itemid] != 0:
                    n_actual_record.append(itemid)
            tmp = tmp[n_actual_record].tolist()
            temp = temp[n_actual_record].tolist()
            for i in range(len(temp)):
                for j in range(i, len(temp)):
                    if (tmp[i] < tmp[j] and temp[i] >= temp[j]) or (tmp[i] > tmp[j] and temp[i] <= temp[j]):
                        n_correct += 1
                        n_all += 1
                    else:
                        n_all += 1

        return n_correct / n_all
    except NameError:
        print("Some Parameter(s) has not been defined yet.")
        print("will return -1 as a result.")
        return -1
    except Exception:   #since lightfm raise Exception('Number of item feature rows does not equal.... ') for the case
        # of Number of item feature (user feature) rows does not equal the number of items (users)
        print("Model has not been fitted to the right dataset yet. It is advised to fit the dataset beforehand and "
              "construct at least the mapping between user and item.")
        print("will return -1 as a result.")
        return -1


def comb(n, k):
    # native on python 3.7 which has not had implementation of math.comb(n, k).
    import math
    if k < 0 or n < 0:
        raise ValueError('Both parameters have to be larger than 0')
    elif k > n:
        raise ValueError('Parameter k={0} is larger than parameter n={1} when k must be smaller than n'.format(k, n))
    else:
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


# test
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import time
# import gc
#
# print("_______________________________________________________")
# print('Tesing on ml-100k')
# userDataset = pd.read_csv(filepath_or_buffer='../ml-100k/u.user', sep='|', header=None, names=['userid', 'age'
#     , 'gender', 'occupation', 'zip'])
# # exercise/recommendationSystem/ml-100k/u.user
# itemDataset = pd.read_csv(filepath_or_buffer='../ml-100k/u.item', sep='|', header=None, names=['itemid'
#     , 'movietitle', 'releasedate', 'videoreleasedate', 'IMDbURL', "unknown", "Action", "Adventure", "Animation"
#     , "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
#                                                                                             "Mystery"
#     , "Romance", "Sci-Fi", "Thriller", "War", "Western"], encoding='ISO-8859-1')
# # exercise/recommendationSystem/ml-100k/u.item
# mappingDataset = pd.read_csv(filepath_or_buffer='exercise/recommendationSystem/ml-100k/u.data', sep='\t', header=None, names=['userid'
#     , 'movieid', 'rating', 'timestamp'])
# # exercise/recommendationSystem/ml-100k/u.data
#
# # convert to list-like
# userInfor = userDataset.values
# itemInfor = itemDataset.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]].values
# mapping = mappingDataset.iloc[:, [0, 1, 2]].values
# mapping_train, mapping_test = train_test_split(mapping, test_size = 0.1, random_state = 0)
#
# # build lightfm dataset
# from lightfm import data
# dataset = data.Dataset()
#
# dataset.fit(users = (userInfor[:, 0]), items = (itemInfor[:, 0]))
#
# # build interaction
# mappingFeatures = dataset.build_interactions(((mappingi[0], mappingi[1], mappingi[2]) for mappingi in mapping))
# mappingFeatures_train = dataset.build_interactions(((mappingi[0], mappingi[1], mappingi[2])
#                                                     for mappingi in mapping_train))
# mappingFeatures_test = dataset.build_interactions(((mappingi[0], mappingi[1], mappingi[2])
#                                                    for mappingi in mapping_test))
#
#
# from lightfm import LightFM
#
# model = LightFM(learning_schedule = 'adagrad', loss='bpr')
# model.fit(mappingFeatures_train[0], sample_weight = mappingFeatures_train[1], epochs = 30, num_threads = 2
#           , verbose = True)
#
# from surprise import NMF
# from surprise import Dataset
# from surprise.model_selection import train_test_split
# from surprise import accuracy
#
# # Load the movielens-100k dataset (download it if needed),
# data = Dataset.load_builtin('ml-100k')
#
# # We'll use the famous SVD algorithm.
# algo = NMF(n_factors = 10, n_epochs = 30, biased = True)
#
# trainset, testset = train_test_split(data, test_size= .1)
#
# algo.fit(trainset)
#
# time_1 = time.time()
# a = fcp(model, mappingFeatures_test)
# time_2 = time.time()
# b = accuracy.fcp(algo.test(testset), verbose = False)
# time_3 = time.time()
# print("Precision: {}".format(abs((a - b) / b)))
# print("Just calculate the time calculate FCP. Does not include time to fit and prepare parameter.")
# print("\tFor my implementation: {}s".format(time_2 - time_1))
# print('\tFor surprise\'s implementation: {}s'.format(time_3 - time_2))
#
# del algo, data, dataset, itemDataset, itemInfor, mapping, mappingDataset, mappingFeatures, mappingFeatures_test\
#     , mappingFeatures_train, mapping_test, mapping_train, model, testset, trainset, userDataset, userInfor
# gc.collect()
#
# print()
# print("_______________________________________________________")
# print('Tesing on ml-20m')
#
# ratingDataset = pd.read_csv(filepath_or_buffer = '../ml-20m/ratings.csv', sep = ',')
# # exercise/recommendationSystem/ml-20m/ratings.csv
# ratingDataset = ratingDataset.iloc[:, [0, 1, 2]]
#
# # convert to list-like
# ratings = ratingDataset.iloc[:, :].values
#
# from surprise import Reader
# read = Reader(rating_scale = (1, 5))
#
# data = Dataset.load_from_df(ratingDataset, read)