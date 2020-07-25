"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

#HTML renders
import codecs

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Home","Visuals","Recommender System"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('first Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.title("We think you'll like:")
                    top_10_movies = ['Interstellar (2014)','Django Unchained (2012)',
                                     'Dark Knight Rises, The (2012)',
                                     'Avengers, The (2012)',
                                     'Guardians of the Galaxy (2014)',
                                     'The Martian (2015)',
                                     'Wolf of Wall Street, The (2013)',
                                     'The Imitation Game (2014)',
                                     'Deadpool (2016)',
                                     'The Hunger Games (2012)']
                    for i,j in enumerate(top_10_movies):
                        st.subheader(str(i+1)+'. '+j)

        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.title("We think you'll like:")
                    top_10_movies = ['Interstellar (2014)','Django Unchained (2012)',
                                     'Dark Knight Rises, The (2012)',
                                     'Avengers, The (2012)',
                                     'Guardians of the Galaxy (2014)',
                                     'The Martian (2015)',
                                     'Wolf of Wall Street, The (2013)',
                                     'The Imitation Game (2014)',
                                     'Deadpool (2016)',
                                     'The Hunger Games (2012)']
                    for i,j in enumerate(top_10_movies):
                        st.subheader(str(i+1)+'. '+j)




    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Home":
        #st.title("Solution Overview")
        #st.write("Describe your winning approach on this page")
        #st.image('resources/imgs/Image_header.png')
        st.write('# Project Building Overview')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        page = codecs.open("resources/HTML/info_page.html", 'r', 'utf-8').read()
        st.markdown(page, unsafe_allow_html=True)
        st.write(" ")
        st.write("# Learn more about the recommenders")
        st.image('resources/imgs/filters.png',use_column_width=True)
        st.write(" ")
        page = codecs.open("resources/HTML/filter_explanation.html", 'r', 'utf-8').read()
        st.markdown(page, unsafe_allow_html=True)

        st.write("# Approach to the winning solution")

        st.write('# Meet the team')
        st.markdown("""| Team member                             | Primary Duty                                                       |
        | :---------------------                | :--------------------                                             |
        | Marcus Moeng                 | Kaggle                           |
        | Thabo Mahlangu | Recommender system model                |
        | Stanley Kobo       | Simple implementation of content-based filtering.                 |
        | Pinky Maredi                    | Trello |
        | Claude Monareng                   | Folder to store model and data binaries if produced.              |
        | Mukovhe Mukwevho                      | Streamlit |""")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


    #--------------------------------------------------------------------
    if page_selection == "Visuals":
        visual = st.sidebar.selectbox("Visual category",
                       ('Movies',
                        'Genres'
                        ))

        if visual =='Movies':
            movie_visual = st.sidebar.radio("select movie visual",
            ('Top 5 Movies',
            'Top 5 Actors',
            'Movie title wordcloud'))

            if movie_visual == 'Movie title wordcloud':
                st.write('# Words used in movie titles')
                st.image('resources/imgs/word_cloud.png')

            elif movie_visual == 'Top 5 Actors':
                st.write("# Top 5 Actors in our Database")
                st.write('## 1. Samuel L. Jackson')
                st.image('resources/imgs/samuel_jackson.jpg',width=10,use_column_width=True)
                st.write('more...')

                st.write('## 2. Steve Buscemi')
                st.image('resources/imgs/steve_B.jpeg',use_column_width=True)

                st.write("## 3. Robert De Niro")
                st.image('resources/imgs/robert.jpg',use_column_width=True)

                st.write("## 4. Nicolas Cage")
                st.image('resources/imgs/cage.jpg',use_column_width=True)

                st.write("## 5. Gerard Depardieu")
                st.image('resources/imgs/gerard.jpg',use_column_width=True)


            elif movie_visual == 'Top 5 Movies':
                st.write("# The best 5 Movies")
                st.write('## 1. Interstellar 2010')
                st.image('resources/imgs/interstellar.jpg',use_column_width=True)

                st.write('## 2. Django Unchained 2012')
                st.image('resources/imgs/django.jpg',use_column_width=True)

                st.write("## 3. Dark Knight Rises 2012")
                st.image('resources/imgs/dark.jpg',use_column_width=True)

                st.write("## 4. Avengers 2012")
                st.image('resources/imgs/avengers.jpg',use_column_width=True)

                st.write("## 5. Guardians of the Galaxy 2014")
                st.image('resources/imgs/galaxy.jpeg',use_column_width=True)


        if visual == "Genres":
            #st.write('# Interesting info on Genres')
            genre_visual = st.sidebar.radio("select genre visual",
            ('Popularity',
            'Runtime',
            'Budget'))

            if genre_visual == 'Popularity':
                st.write('# Genre popularity')
                st.image('resources/imgs/genres.png',use_column_width=True)

                st.image('resources/imgs/tree_map.png',use_column_width=True)

            elif genre_visual == 'Runtime':
                st.image('resources/imgs/genre_runtime.png',use_column_width=True)

            else:
                st.image('resources/imgs/genre_budget.png',use_column_width=True)





if __name__ == '__main__':
    main()
