# Reginald_MaameYaa_Intro_to_AI_Final_Project
# README FILE FOR MOVIE RECOMMENDATION SYSTEM

## Movie Recommendation System
===========================

## Overview
--------
This project develops a personalized movie recommendation system using K-Nearest Neighbors (KNN). 
The system recommends movies based on user-selected titles, leveraging movie ratings and features for accurate suggestions.

# Dataset
-------
Columns:'Poster_Link', 'Series_Title', 'Released_Year', 'Certificate', 'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross'
The columns that the dataset primarily used for feature calculation were: Genre, Released_Year, Certificate, Director, Star1, Star2, Star3, Star4, No_of_Votes


## Project Structure
-----------------
1. Data Preprocessing: Handle missing values, encode categorical features, and standardize numerical features.
2. Feature Engineering: Convert and handle 'Released_Year' and select relevant features.
3. Model Training and Evaluation: Use KNeighborsRegressor with GridSearchCV for tuning. Validate and test the model.
4. Movie Recommendation: Function to find closest 10 movies based on selected title.

## How to Run
----------
1. Install necessary libraries:
    pip install pandas numpy scikit-learn
2. Load dataset: Ensure the dataset is in CSV format (e.g., 'movie_ratings.csv').
3. Run the script or notebook:
    - 'Reggie.py': Complete Python script.
    - 'Final Project.ipynb': Jupyter Notebook with step-by-step implementation.

## Example Usage
-------------
1. Load the dataset:
    df = pd.read_csv('movie_ratings.csv')
2. Get recommendations:
    selected_movie = 'The Shawshank Redemption'
    recommended_movies = recommend_movies(selected_movie, best_knn, n_recommendations=10)
    print(f"Movies similar to '{selected_movie}':")
    print(recommended_movies)


## How the Application Works
-------------
1. Scroll and select a previously watched movie from the list.
2. The system checks and compares all the movies with features similar to those selected.
3. The system recommends the best 10 movies in relation to what the user selected.
4. The user can download the list of movies provided.
5. The user can search for a movie that he or she presumes will be on the list rather than having to scroll to check if the movie is on the list.
6. The user can expand the list.

Link:
