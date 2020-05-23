import pandas as pd
from math import sqrt 
import numpy as np
import matplotlib.pyplot as plt 

# Preprocessing

#Storing the movie information into a pandas dataframe
movies_df = pd.read_csv(r'moviedataset\ml-latest\movies.csv')
#Storing the user information into a pandas dataframe
ratings_df = pd.read_csv(r'moviedataset\ml-latest\ratings.csv')
#Head is a function that gets the first N rows of a dataframe. N's default is 5.
movies_df.head()

#Removing year from movie title column
#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df.head()

#Now we split values in Genres column into  a list.
#Every genre is separated by a | so we simply have to call the split function on |
movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()


#Converting genres from list 0/1 vectors
#Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df = movies_df.copy()

#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column

#Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df = movies_df.copy()

#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()


#Preprocessing ratings DataFrame
ratings_df.head()
ratings_df = ratings_df.drop('timestamp', 1)
ratings_df.head()

# CONTENT-BASED RECOMMENDATION SYSTEM

userInput = [
            {'title':'Pianist, The', 'rating':4.5},
            {'title':'Good Will Hunting', 'rating':5},
            {'title':'Bourne Ultimatum, The', 'rating':4.5},
            {'title':"Pulp Fiction", 'rating':4},
            {'title':"Toy Story 3", 'rating':2},
            {'title':"Furious 7", 'rating':2.5},
            {'title':"Fast Five (Fast and the Furious 5, The)", 'rating':2.5},
            {'title':'Beautiful Mind, A', 'rating':5},
            {'title':'Saving Private Ryan', 'rating':5},
            {'title':'Prestige, The', 'rating':5},
            {'title':'I Am Legend', 'rating':4},
            {'title':'Black Panther', 'rating':4},
            {'title':'Inception', 'rating':5},
            {'title':'Django Unchained', 'rating':4},
            {'title':'Top Gun', 'rating':4},
            {'title':'Slumdog Millionaire', 'rating':5},
            {'title':'Dark Knight, The', 'rating':5},
            {'title':'Godfather, The', 'rating':5},
            {'title':'Godfather: Part II, The', 'rating':5},
            {'title':'RoboCop', 'rating':2},
            {'title':'RoboCop 3', 'rating':2},
            
            
         ] 
inputMovies = pd.DataFrame(userInput)
inputMovies

#Adding Movieid to input user

#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
inputMovies

#Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies

#Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
#Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
userGenreTable

inputMovies['rating']

#Dot produt to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
#The user profile
userProfile



#Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
#And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
genreTable.head()
genreTable.shape
genreTable

#Multiply the genres by the weights and then take the weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()


#Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
#Just a peek at the values
recommendationTable_df.head()

pd.set_option('display.max_columns', None)

#The final recommendation table
movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]




