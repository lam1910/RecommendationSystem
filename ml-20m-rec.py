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

import pandas as pd
import numpy as np
import gc

itemDataset = pd.read_csv(filepath_or_buffer = 'ml-20m/movies.csv', sep = ',')
# exercise/recommendationSystem/ml-20m/movies.csv
ratingDataset = pd.read_csv(filepath_or_buffer = 'ml-20m/ratings.csv', sep = ',')
# exercise/recommendationSystem/ml-20m/ratings.csv
tagScoreDataset = pd.read_csv(filepath_or_buffer = 'ml-20m/processed_genome_score.csv', sep = ',')
# exercise/recommendationSystem/ml-20m/processed_genome_score.csv
# tmpDataset just to keep tab of the content of each tag. If weak machine, can delete this for memory purpose
tmpDataset = pd.read_csv(filepath_or_buffer = 'ml-20m/genome-tags.csv', sep = ',')
# exercise/recommendationSystem/ml-20m/genome-tags.csv