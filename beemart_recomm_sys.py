# dataset will not be published on github and only available on local machine. if you want to you this file, you will
# need to use your own dataset. There will be a walkthrough on how the structure of the files are, so you can use your
# own file following that structure or create new ones or change how we read and process the data.

import pandas as pd
import os
import gc
from sklearn.model_selection import train_test_split

def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

#importing dataset
interaction = pd.read_excel(io = '20191113_Khachhang_Product_Beemart/order_khachang_sp.xlsx')
# exercise/recommendationSystem/20191113_Khachhang_Product_Beemart/order_khachang_sp.xlsx
a = interaction.shape[0]
print('Original number of row: %s' %a)

print()
print('______________________________________________')
# remove customer under alias
print('Removing customers that are not recorded by name in dataset.')
alias = ['kh', 'KH', 'Kh', 'Khách lẻ']
row_drop = interaction.loc[interaction.CustomerName.isin(alias)].index
interaction.drop(row_drop, 0, inplace = True)
interaction.reset_index(drop = True, inplace = True)
print('Row Deleted: %s' %(a - interaction.shape[0]))
no_del = a - interaction.shape[0]
a = interaction.shape[0]

print()
print('______________________________________________')
# remove customer with interaction < 10 times
print('Removing customer with interaction <= 10 times')
clean_interaction = interaction.groupby('CustomerId').count()
id_to_del = clean_interaction.loc[clean_interaction.amount <= 10].index.tolist()
row_drop = interaction.loc[interaction.CustomerId.isin(id_to_del)].index
interaction.drop(row_drop, 0, inplace = True)
interaction.reset_index(drop = True, inplace = True)
print('Row Deleted: %s' %(a - interaction.shape[0]))
print('\tRow Deleted in total: %s' %(no_del + a - interaction.shape[0]))
print()
print('______________________________________________')
print('Row remained: %s' %interaction.shape[0])

# write out new interaction
if len(find_all('interaction.xlsx', '20191113_Khachhang_Product_Beemart')) == 0:
    # exercise/recommendationSystem/20191113_Khachhang_Product_Beemart
    interaction.to_excel(excel_writer = '20191113_Khachhang_Product_Beemart/interaction.xlsx', index = False)
    # exercise/recommendationSystem/20191113_Khachhang_Product_Beemart/interaction.xlsx
else:
    print('File was already created and ready to use. If you really want to rewrite the file, copy the if part to '
          'console to run it.')

print()
print('______________________________________________')
print('Loading the update interaction. Removing unnecessary columns')
# interaction = pd.read_excel(io = '20191113_Khachhang_Product_Beemart/interaction.xlsx')
# exercise/recommendationSystem/20191113_Khachhang_Product_Beemart/interaction.xlsx

del a, alias, clean_interaction, id_to_del, no_del, row_drop
gc.collect()

# remove unnecessary column
interaction = interaction.iloc[:, [1, 3, 5, 6]]

# create interaction with rating as number of product that the customer buy
by_no_product = interaction.groupby(['CustomerId', 'ProductId'], sort = False).sum().so_luong.reset_index(drop = False)

def rescaling(dataset, column, list_of_ms):
    # Rescaling the value in dataset[column] to 0, 1, ... 5 based on [min, n1), [n1, n2), [n2, n3), [n3, n4)
    # and [n_n, max]
    if not isinstance(dataset, pd.DataFrame):
        print('Not the right object on param dataset.')
        raise TypeError('Param 1 must be of pandas.DataFrame type')
    elif not isinstance(column, str):
        print('Not the right object on param column.')
        raise TypeError('Param 2 must be of string type')
    elif not isinstance(list_of_ms, list):
        print('Not the right object on param list_of_ms.')
        raise TypeError('Param 3 must be of list type')
    elif False in [(item != True and item != False) and (isinstance(item, float) or isinstance(item, int))
                 for item in list_of_ms]:
        print('Not the right object type for elements of param list_of_ms.')
        stack = [(item != True and item != False) and (isinstance(item, float) or isinstance(item, int))
                 for item in list_of_ms]
        raise TypeError('Number of element that is not a real number: %s' %stack.count(False))
    elif len(list_of_ms) != 4:
        print('Not the right number of element for param list_of_ms')
        raise IndexError('Param 3 must contains 4 elements.')
    else:
        try:
            values = dataset[column].tolist()
            returnV = []
            list_of_ms.sort()
            for element in values:
                if element < list_of_ms[0]:
                    returnV.append(1)
                elif element < list_of_ms[1]:
                    returnV.append(2)
                elif element < list_of_ms[2]:
                    returnV.append(3)
                elif element < list_of_ms[3]:
                    returnV.append(4)
                else:
                    returnV.append(5)
            dataset[column] = returnV
        except KeyError:
            print('Cannot find key %s in the dataframe. Nothing will change in the dataframe.' %column)

# number based on real life observation of the datasets. Must change each time making any changes the dataset
# or using a new one.
rescaling(by_no_product, 'so_luong', [2, 10, 100, 1000])

# rename to rating
by_no_product.rename(columns = {'so_luong': 'rating'}, inplace = True)

all_customer = by_no_product.CustomerId.unique()
all_product = by_no_product.ProductId.unique()

# splitting dataset
no_prod_train, no_prod_test = train_test_split(by_no_product.values, test_size = 0.2, random_state = 0)

# build lightfm dataset

from lightfm import data
print()
print('______________________________________________')
print('Building dataset using item count as rating.')
dataset_item = data.Dataset()
dataset_item.fit(users = all_customer[:], items = all_product[:])

# build mapping
mappingProd_train = dataset_item.build_interactions(((mappingi[0], mappingi[1], mappingi[2])
                                                    for mappingi in no_prod_train))
mappingProd_test = dataset_item.build_interactions(((mappingi[0], mappingi[1], mappingi[2])
                                                   for mappingi in no_prod_test))


from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

print()
print('______________________________________________')
print('Training using item count as rating.')
model_prod = LightFM(learning_schedule = 'adagrad', loss='bpr')
model_prod.fit(mappingProd_train[0], sample_weight = mappingProd_train[1], epochs = 30, num_threads = 2, verbose = True)

print("Train precision at 3rd: %.4f"
      % precision_at_k(model_prod, mappingProd_train[0], k = 3).mean())
print("Test precision at 3rd: %.4f"
      % precision_at_k(model_prod, mappingProd_test[0], k = 3).mean())

print("ROC AUC metric for train: %.4f"
      % auc_score(model_prod, mappingProd_train[0]).mean())
print("ROC AUC metric for test: %.4f"
      % auc_score(model_prod, mappingProd_test[0]).mean())

import accuracy

print('FCP of train set: {0}'.format(accuracy.fcp(model_prod, mappingProd_train)))

print('FCP of test set: {0}'.format(accuracy.fcp(model_prod, mappingProd_test)))
