import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and preprocessing steps
best_knn = joblib.load('best_knn_model.joblib')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')

# Load the dataset
df = pd.read_csv('dataset2names.csv', low_memory=False, na_values='-')


# Function to preprocess the input data
def preprocess_input(df, features):
    df['Gross'] = df['Gross'].str.replace(',', '').astype(float)
    df['IMDB_Rating'] = df['IMDB_Rating'].fillna(df['IMDB_Rating'].mode()[0])
    df['Meta_score'] = df['Meta_score'].fillna(df['Meta_score'].mode()[0])
    df['Gross'] = df['Gross'].fillna(df['Gross'].mode()[0])
    df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
    df['Released_Year'] = df['Released_Year'].fillna(df['Released_Year'].mode()[0])

    for col in features:
        if df[col].dtype == 'object':
            df[col] = encoder.fit_transform(df[col])

    return df


# Preprocess the dataset
features = ['Genre', 'Released_Year', 'Certificate', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes']
df = preprocess_input(df, features)
X = df[features]
X_scaled = scaler.transform(X)


# Function to recommend movies
def recommend_movies(movie_title, knn_model, n_recommendations=10):
    # Find the index of the movie
    movie_idx = df[df['Series_Title'] == movie_title].index[0]

    # Get the feature vector of the selected movie
    movie_features = X_scaled[movie_idx].reshape(1, -1)

    # Find the K nearest neighbors (excluding the movie itself)
    distances, indices = knn_model.kneighbors(movie_features, n_neighbors=n_recommendations + 1)

    # Get the recommended movie indices (excluding the movie itself)
    recommended_indices = indices.flatten()[1:]

    # Create a DataFrame for recommended movies
    recommendations = df.iloc[recommended_indices][['Series_Title', 'IMDB_Rating']]

    return recommendations


# Streamlit app
st.title('Movie Recommendation System')

selected_movie = st.selectbox('Select a movie', df['Series_Title'].unique())

if st.button('Recommend Movies'):
    recommended_movies = recommend_movies(selected_movie, best_knn)
    st.write(f"Movies similar to '{selected_movie}':")
    st.dataframe(recommended_movies)
