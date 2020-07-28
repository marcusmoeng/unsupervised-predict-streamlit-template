"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import random
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer



# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',',delimiter=',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/Base_SVD.pkl', 'rb'))

def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    # Data preprosessing
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df,reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
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



    indices = pd.Series(movies_df['title'])
    movie_ids = pred_movies(movie_list)
    df_init_users = ratings_df[ratings_df['userId']==movie_ids[0]]
    for i in movie_ids :
        df_init_users=df_init_users.append(ratings_df[ratings_df['userId']==i])


    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]

    num = int(str(idx_1)+str(idx_2)+str(idx_3))
    random.seed(num)
    ratings = movies_df.merge(ratings_df, on='movieId')


    # Creating a Series with the similarity scores in descending order
    #rank_1 = cosine_sim[idx_1]
    #rank_2 = cosine_sim[idx_2]
    #rank_3 = cosine_sim[idx_3]
    #df_init_users = df_init_users.sort_values(['rating'], ascending=False)

    #top_10_movies = list(df_init_users['movieId'].head(10))

    #top10=movies_df['movieId'].isin(top_10_movies)
    #recommended_movies = list(movies_df[top10]['title'])

    #for movie_index in top_10_movies:
    #top10=movies_df['movieId'].isin(top_10_movies)


    #similar = get_similar(idx_1,5)
    userId_list = list(df_init_users['userId'])
    similar_users = ratings[ratings['userId'].isin(userId_list)].sort_values('rating',ascending=False)
    similar_users = similar_users[similar_users['rating']>=3]
    recommended_movies = random.choices(list(similar_users['title']),k=10)
    #print(ratings[ratings['movieId'].isin(list(df_init_users['movieId']))])
    #print(corrMatrix)

    """
    listing_list = []
    index_list = list([idx_1,idx_2,idx_3])

    for idx in index_list:
        try:
            rank = cosine_sim[idx]
            score_series = pd.Series(rank).sort_values(ascending = False)
            listing_list.append(score_series)
        except:
            pass


    print(listing_list)
    print("listing ended")
    if len(listing_list)==1:
        listings = listing_list[0]

        recommended_movies = []
        # Appending the names of movies
        top_50_indexes = list(listings.iloc[1:50].index)
        # Removing chosen movies
        top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
        for i in top_indexes[:top_n]:
            recommended_movies.append(list(movies['title'])[i])
        return recommended_movies

    elif len(listing_list)>1:
        print('I get here')
        listings = listing_list[0]
        for listing_entry in listing_list[1:]:
            listings.append(listing_entry)

        listings = listings.sort_values(ascending=True)

        recommended_movies = []
        # Appending the names of movies
        top_50_indexes = list(listings.iloc[1:50].index)
        # Removing chosen movies
        top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
        for i in top_indexes[:top_n]:
            recommended_movies.append(list(movies['title'])[i])
        return recommended_movies


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


    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
     # Appending the names of movies
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)
    recommended_movies = []
    # Choose top 50
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies_df['title'])[i])
    """


    return recommended_movies
