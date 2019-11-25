# read data
"""
Summary
=======
This dataset (ml-20m) describes 5-star rating and free-text tagging activity from [MovieLens]
(http://movielens.org), a movie recommendation service. It contains 20000263 ratings and 465564 tag
applications across 27278 movies. These data were created by 138493 users between January 09, 1995
and March 31, 2015. This dataset was generated on March 31, 2015, and updated on October 17, 2016 to update
links.csv and add genome-* files.
Users were selected at random for inclusion. All selected users had rated at least 20 movies. No
demographic information is included. Each user is represented by an id, and no other information is
provided.
The data are contained in six files, `genome-scores.csv`, `genome-tags.csv`, `links.csv`, `movies.csv`
, `ratings.csv` and `tags.csv`. More details about the contents and use of all these files follows.
This and other GroupLens data sets are publicly available for download at <http://grouplens.org/datasets/>.

Citation
========
To acknowledge use of the dataset in publications, please cite the following paper:
> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM
Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
DOI=<http://dx.doi.org/10.1145/2827872>

Content and Use of Files
========================
Formatting and Encoding
-----------------------
The dataset files are written as
[comma-separated values](http://en.wikipedia.org/wiki/Comma-separated_values) files with a single header
row. Columns that contain commas (`,`) are escaped using double-quotes (`"`). These files are encoded as
UTF-8. If accented characters in movie titles or tag values (e.g. Misérables, Les (1995)) display
incorrectly, make sure that any program reading the data, such as a text editor, terminal, or script,
is configured for UTF-8.

User Ids
--------
MovieLens users were selected at random for inclusion. Their ids have been anonymized. User ids are
consistent between `ratings.csv` and `tags.csv` (i.e., the same id refers to the same user across the
two files).

Movie Ids
---------
Only movies with at least one rating or tag are included in the dataset. These movie ids are consistent
with those used on the MovieLens web site (e.g., id `1` corresponds to the URL
<https://movielens.org/movies/1>).
Movie ids are consistent between `ratings.csv`, `tags.csv`, `movies.csv`, and `links.csv` (i.e., the
same id refers to the same movie across these four data files).


Ratings Data File Structure (ratings.csv)
-----------------------------------------
All ratings are contained in the file `ratings.csv`. Each line of this file after the header row
represents one rating of one movie by one user, and has the following format:

    userId,movieId,rating,timestamp

The lines within this file are ordered first by userId, then, within user, by movieId.
Ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).
Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

Tags Data File Structure (tags.csv)
-----------------------------------
All tags are contained in the file `tags.csv`. Each line of this file after the header row represents
one tag applied to one movie by one user, and has the following format:

    userId,movieId,tag,timestamp

The lines within this file are ordered first by userId, then, within user, by movieId.
Tags are user-generated metadata about movies. Each tag is typically a single word or short phrase. The
meaning, value, and purpose of a particular tag is determined by each user.
Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

Movies Data File Structure (movies.csv)
---------------------------------------
Movie information is contained in the file `movies.csv`. Each line of this file after the header row
represents one movie, and has the following format:

    movieId,title,genres

Movie titles are entered manually or imported from <https://www.themoviedb.org/>, and include the year
of release in parentheses. Errors and inconsistencies may exist in these titles.
Genres are a pipe-separated list, and are selected from the following:

    * Action
    * Adventure
    * Animation
    * Children's
    * Comedy
    * Crime
    * Documentary
    * Drama
    * Fantasy
    * Film-Noir
    * Horror
    * Musical
    * Mystery
    * Romance
    * Sci-Fi
    * Thriller
    * War
    * Western
    * (no genres listed)

Links Data File Structure (links.csv)
---------------------------------------
Identifiers that can be used to link to other sources of movie data are contained in the file `links.csv`.
Each line of this file after the header row represents one movie, and has the following format:

    movieId,imdbId,tmdbId

movieId is an identifier for movies used by <https://movielens.org>. E.g., the movie Toy Story has the
link <https://movielens.org/movies/1>.
imdbId is an identifier for movies used by <http://www.imdb.com>. E.g., the movie Toy Story has the link \
<http://www.imdb.com/title/tt0114709/>.
tmdbId is an identifier for movies used by <https://www.themoviedb.org>. E.g., the movie Toy Story has
the link <https://www.themoviedb.org/movie/862>.
Use of the resources listed above is subject to the terms of each provider.

Tag Genome (genome-scores.csv and genome-tags.csv)
-------------------------------------------------
This data set includes a current copy of the Tag Genome.
[genome-paper]: http://files.grouplens.org/papers/tag_genome.pdf
The tag genome is a data structure that contains tag relevance scores for movies.  The structure is a
dense matrix: each movie in the genome has a value for *every* tag in the genome.
As described in [this article][genome-paper], the tag genome encodes how strongly movies exhibit
particular properties represented by tags (atmospheric, thought-provoking, realistic, etc.). The tag
genome was computed using a machine learning algorithm on user-contributed content including tags,
ratings, and textual reviews.
The genome is split into two files.  The file `genome-scores.csv` contains movie-tag relevance data in the
following format:

    movieId,tagId,relevance

The second file, `genome-tags.csv`, provides the tag descriptions for the tag IDs in the genome file, in
the following format:

    tagId,tag

The `tagId` values are generated when the data set is exported, so they may vary from version to version
of the MovieLens data sets.

Cross-Validation
----------------
Prior versions of the MovieLens dataset included either pre-computed cross-folds or scripts to perform
this computation. We no longer bundle either of these features with the dataset, since most modern
toolkits provide this as a built-in feature. If you wish to learn about standard approaches to cross-fold
computation in the context of recommender systems evaluation, see [LensKit](http://lenskit.org) for tools,
documentation, and open-source code examples.
"""

"""
Focus solely on gnome tag and content-based filtering. Will not need rating thus can be consider as a cold-start problem
Solution follow the instruction on Mohd Ali, Syed & Nayak, Dr & Barik, Dr. Rabindra & Lenka, Rakesh. (2017). Movie 
Recommendation System using Genome Tags and Content-based Filtering.
https://www.researchgate.net/publication/321579366_Movie_Recommendation_System_using_Genome_Tags_and_Content-based_Filtering
Recommendation system has become of utmost  importance during the last decade. It is due to the fact that a good 
recommender system can help assist people in their decision making process on the daily basis. When it comes to movie, 
collaborative recommendation tries to assist the users by using help of similar type of users or movies from their 
common historical ratings. Genre is one of the major meta tag used to classify similar type of movies, as these genre 
are binary in nature they might not be the best way to recommend. In this paper, a hybrid model is proposed which 
utilizes genomic tags of movie coupled with the content- based filtering to recommend similar movies. It uses Principal 
Component Analysis (PCA) and pearson correlation techniques to reduce the tags which are redundant and show low 
proportion of variance, hence reducing the computation complexity. Initial results prove that genomic tags gives the 
better result in terms of finding similar type of movies, gives more accurate and personalized recommendation as 
compared to existing models.
"""

import pandas as pd
import numpy as np
import gc
import warnings

print("Due to the fact that there are more movies in the itemDataset than in tagScoreDataset (itemDataset.shape[0] > "
      "len(tagScoreDataset.movieId.unique().tolist), or simply put there are some movies in the database that have not "
      "been given a tag yet, we will need a list of Id of all movies that had not had a tag yet to build the predict "
      "(recommend) function.")
itemDataset = pd.read_csv(filepath_or_buffer = 'ml-20m/movies.csv', sep = ',')
# exercise/recommendationSystem/ml-20m/movies.csv

tagScoreDataset = pd.read_csv(filepath_or_buffer = 'ml-20m/genome-scores.csv', sep = ',')
# exercise/recommendationSystem/ml-20m/genome-scores.csv
# tmpDataset just to keep tab of the content of each tag. If weak machine, can delete this for memory purpose
tmpDataset = pd.read_csv(filepath_or_buffer = 'ml-20m/genome-tags.csv', sep = ',')
# exercise/recommendationSystem/ml-20m/genome-tags.csv

message = "Due to the fact that movieId is in type of integer, dtype in series.astype = int. In other problem, might " \
          "want to change it to str"
warnings.warn(message, stacklevel = 1)
items = tagScoreDataset.movieId.astype(int).unique().tolist()
rows_of_array = []
print("Adding the movieId at the beginning of each row for printing out the processed data as "
      "processed_genome_score.csv. If do not need that file, you could delete the row.insert(0, item) from the next "
      "loop. Opting to delete or not will not affect how the program work but it will change the content of the output "
      "file.")
for item in items:
    row = tagScoreDataset.loc[tagScoreDataset.movieId == item].relevance.tolist()
    row.insert(0, item)
    rows_of_array.append(row)

# matrix of genome score average
processed_data = np.array(rows_of_array)

col_name = ['movieId']
for i in range(1, 1129):
    col_name.append(str(i))
#write out from column 1 to 1128 (sometimes from 0 to 1127) will not have specific name since they are the weight of one
# tag from tagId 1 - 1128 for movieId that in index + 1.
try:
    output = pd.DataFrame(data = processed_data, columns = col_name)
except ValueError:
    del col_name[0]
    output = pd.DataFrame(data = processed_data, columns = col_name)

output.to_csv(path_or_buf = 'ml-20m/processed_genome_score.csv', index = False)
# exercise/recommendationSystem/ml-20m/processed_genome_score.csv

# filter to get the matrix of genome score average
retain = list(range(1, 1129))
try:
    processed_data = output.iloc[:, retain].values
except IndexError:
    message = 'process_data do not contain movie Id, safe to use without the need of filtering'
    warnings.warn(message, stacklevel = 1)
finally:
    del retain, output
    gc.collect()

items_not_tagged = []
print("Again since movieId is in type of integer, dtype in series.astype = int. In other problem, might want "
      "to change it to str")
all_items = itemDataset.movieId.astype(int).unique().tolist()
for item in all_items:
    if item not in items:
        items_not_tagged.append(item)


# perform PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.9, whiten = True, svd_solver = 'full')

processed_data = pca.fit_transform(processed_data)

# next step:
#   split the table into each row (a vector of a movie)
#   computer the cosine of the angle between each vector obtain an 10381 * 10381 matrix (in paper they select 2000)
#   build a recommend function work as follow:
#       input an movie id
#       if input (say n) is in the items_not_tagged list raise an error movie has no tag
#       elif n in items get index of n in the list (say k), go to kth column in the matrix just obtain, get 5 biggest
#           value than transfer back to movieId for those index
#       else raise error movieId not found
#       output the array contain 5 nearest movieId.
# already have rows_of_array with each element is the movieId and vector of the movie

message = 'Main Part of the program. Extremely compute-intensive. Can be reduce to a 5000 * 5000 size array. Note to ' \
          'change items list and items_not_tagged list'
warnings.warn(message, stacklevel = 1)

length_of_vector = np.linalg.norm(x = processed_data, axis = 1)

del all_items
gc.collect()

matrix_of_cos = np.ones(shape = (10381, 10381), dtype = float)

for i in range(10381):
    for j in range(i + 1, 10381):
        matrix_of_cos[i, j] = \
            np.dot(a = processed_data[i, :], b = processed_data[j, :]) / (length_of_vector[i] * length_of_vector[j])
        matrix_of_cos[j, i] = matrix_of_cos[i, j]

# func to recommend the item based solely on genome-tag
# input: n: Id of the item you have (could be str or int)
# output: an array of five integer represented five item id that are closest to the item n
#         return None if cannot convert n into int
# errors: ValueError('Item is not tagged yet') for item that does not have any tag
#         ValueError('Item is not on the database') for item that is not in the dataset

def recommend(n):
    try:
        input = int(n)
    except ValueError:
        message = 'Cannot recognize input. Will return None'
        warnings.warn(message, SyntaxWarning, stacklevel = 1)
        return None

    if input in items:
        result = []
        movieIndex = items.index(input)
        values_of_cos = pd.DataFrame(matrix_of_cos[:, movieIndex])
        values_of_cos.sort_values(by = 0, ascending = False, inplace = True)
        neighborIndex = values_of_cos.index[1:6]
        for id in neighborIndex:
            result.append(items[id])
        return result
    elif input in items_not_tagged:
        raise ValueError('Item is not tagged yet')
    else:
        raise ValueError('Item is not on the database')

# func to get the movie information inside the itemDataset
# input: id: Id of the item you have (could be str or int)
# output: a string that has the format of 'MovieId: %s: %s, (%s). %{movieId, title, (all the genres)}'
#         return None if cannot convert n into int
# errors: ValueError('Item is not on the database') for item that is not in the dataset
def getMovieInfor(id):
    try:
        input = int(id)
    except ValueError:
        message = 'Cannot recognize input. Will return None'
        warnings.warn(message, SyntaxWarning, stacklevel = 1)
        return None

    if input in items or input in items_not_tagged:
        movieInfor = itemDataset.loc[itemDataset.movieId == input]
        genres = movieInfor.iloc[:, 2].values[0].split('|')
        result = 'MovieId: ' + str(movieInfor.iloc[:, 0].values[0]) + ': '
        result += movieInfor.iloc[:, 1].values[0]
        for genre in genres:
            result += ', '
            result += genre

        result += '.'
        return result
    else:
        raise ValueError('Item is not on the database')

listOfMovie = recommend(5)
print('Movie: ' + getMovieInfor(5))
print('You might also like to watch:')
for id in range(len(listOfMovie)):
    print('\t{}. '.format(id + 1) + getMovieInfor(listOfMovie[id]))
