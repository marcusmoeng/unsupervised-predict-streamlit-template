"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',',delimiter=',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # Split genre data into individual words.
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    # Subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Initializing the empty list of recommended movies
    recommended_movies = []
    data = data_preprocessing(5000)
    # Instantiating and generating the count matrix
    count_vec = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    count_matrix = count_vec.fit_transform(data['keyWords'])
    indices = pd.Series(data['title'])
    cosine_sim = linear_kernel(count_matrix, count_matrix)


    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]

    ##
    idx_list = [idx_1,idx_2,idx_3]
    listings_list = []

    for index in idx_list:
        try:
            rank = cosine_sim[index]
            score_series = pd.Series(rank).sort_values(ascending = False)
            listings_list.append(score_series)

        except:
            pass

    if len(listings_list) >1:
        listing = listings_list[0]
        for i in range(1,len(listings_list)):

            listing.append(listings_list[i])


        #listings = listings.sort_values(ascending=False)

        # Store movie names
        recommended_movies = []
        # Appending the names of movies
        top_50_indexes = list(listing.iloc[1:50].index)
        # Removing chosen movies
        top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
        for i in top_indexes[:top_n]:
            recommended_movies.append(list(movies['title'])[i])


        elif len(listings_list)==1:
            listing = listings_list[0]

            # Store movie names
            recommended_movies = []
            # Appending the names of movies
            top_50_indexes = list(listing.iloc[1:50].index)
            # Removing chosen movies
            top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
        for i in top_indexes[:top_n]:
            recommended_movies.append(list(movies['title'])[i])

    else:
        recommended_movies = ['Interstellar (2014)','Django Unchained (2012)',
                     'Dark Knight Rises, The (2012)',
                     'Avengers, The (2012)',
                     'Guardians of the Galaxy (2014)',
                     'The Martian (2015)',
                     'Wolf of Wall Street, The (2013)',
                     'The Imitation Game (2014)',
                     'Deadpool (2016)',
                     'The Hunger Games (2012)']


    return recommended_movies
