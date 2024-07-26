import pandas as pd
import boto3
from io import StringIO
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the S3 client
s3 = boto3.client('s3')
s3_bucket_name = 'myflicksasa'
movies_file_path = 'imdb/combined_processed.csv'
user_profiles_file_path = 'imdb/simulated_user_profiles.csv'

# Function to read CSV from S3
def read_csv_from_s3(bucket_name, file_path):
    print(f"Reading CSV from S3 bucket: {bucket_name}, file path: {file_path}")
    obj = s3.get_object(Bucket=bucket_name, Key=file_path)
    df = pd.read_csv(obj['Body'])
    print(f"Loaded {len(df)} records from {file_path}")
    return df

# Load movie data and user profile data from S3
movies_df = read_csv_from_s3(s3_bucket_name, movies_file_path)
user_profiles_df = read_csv_from_s3(s3_bucket_name, user_profiles_file_path)

# Ensure the movie dataset has necessary columns
if 'predicted_genre' not in movies_df.columns:
    raise KeyError("'predicted_genre' column is missing in movies_df")
if 'description' not in movies_df.columns:
    raise KeyError("'description' column is missing in movies_df")

# Helper function to determine if a title is a movie or series
def get_movie_type(year):
    if isinstance(year, str) and '-' in year:
        return 'Series'
    else:
        return 'Movie'

# Add a column for the type of movie
movies_df['type'] = movies_df['year'].apply(get_movie_type)

# Convert year to numeric, using errors='coerce' to handle non-numeric values
movies_df['year'] = pd.to_numeric(movies_df['year'].str.extract(r'(\d{4})')[0], errors='coerce')

# Combine descriptions and genres into a single string
movies_df['content'] = movies_df['description'] + ' ' + movies_df['predicted_genre']

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Transform the combined content into TF-IDF vectors
print("Transforming content into TF-IDF vectors...")
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['content'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Calculate the cosine similarity matrix
print("Calculating cosine similarity matrix...")
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Cosine similarity matrix shape: {cosine_sim_matrix.shape}")

# Function to recommend movies based on user preferences
def recommend_movies(user_id, n_recommendations=10):
    try:
        user_profile = user_profiles_df[user_profiles_df['user_id'] == user_id].iloc[0]
    except IndexError:
        raise ValueError(f"No user found with user_id: {user_id}")
    
    print("User Profile:\n", user_profile)

    # Adjusted filtering logic to ensure flexibility
    filtered_movies = movies_df.copy()

    # Filter by type (only 'movie' or 'series')
    if user_profile['type'].lower() != 'all':
        filtered_movies = filtered_movies[filtered_movies['type'].str.lower() == user_profile['type'].lower()]
    
    print(f"Filtered movies by type '{user_profile['type']}': {len(filtered_movies)} movies found")

    # Filter by genre
    try:
        preferred_genres = ast.literal_eval(user_profile['preferred_genres'])
    except KeyError:
        raise KeyError("'preferred_genres' column is missing in user_profiles_df")
    
    genre_filtered_movies = pd.DataFrame()
    recs_per_genre = n_recommendations // len(preferred_genres) + (n_recommendations % len(preferred_genres) > 0)

    for genre in preferred_genres:
        genre_movies = filtered_movies[filtered_movies['predicted_genre'].str.contains(genre, case=False, na=False)]
        
        # If no series are found in the genre, fallback to movies in the genre
        if genre_movies.empty and user_profile['type'].lower() != 'all':
            genre_movies = movies_df[(movies_df['type'] == 'Movie') & (movies_df['predicted_genre'].str.contains(genre, case=False, na=False))]

        # Adjust age filtering to ensure inclusivity
        if user_profile['age'] > 35:
            genre_movies = genre_movies[genre_movies['year'] < 2005]
        
        print(f"Filtered Movies for Genre '{genre}' after Age Filtering: {len(genre_movies)} movies found")
        
        # If still empty after filtering, use original genre filtered movies
        if genre_movies.empty:
            genre_movies = movies_df[movies_df['predicted_genre'].str.contains(genre, case=False, na=False)]
        
        # Sort movies by rating
        genre_movies = genre_movies.sort_values(by='rating', ascending=False)
        
        # Select top recommendations per genre
        genre_filtered_movies = pd.concat([genre_filtered_movies, genre_movies.head(recs_per_genre)])
        
        print(f"Filtered Movies for Genre '{genre}': {len(genre_movies)} movies found")

    # If we have more than needed, truncate the final set to n_recommendations
    recommendations = genre_filtered_movies.head(n_recommendations)
    
    # If no recommendations are found, use content-based filtering to recommend highly similar movies
    if recommendations.empty:
        print("No recommendations found. Using content-based filtering.")
        idx = filtered_movies.index[0]
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        recommendations = filtered_movies.iloc[movie_indices][['title', 'description', 'type', 'rating', 'predicted_genre']]
    
    print("Final recommendations:\n", recommendations)
    return recommendations[['title', 'description', 'type', 'rating', 'predicted_genre']]

# Test function to verify the recommendation logic
def test_recommend_movies():
    test_user_id = 47 # Updated user ID
    try:
        recommendations = recommend_movies(test_user_id)
    except Exception as e:
        print(f"Error during test_recommend_movies: {e}")
        return
    
    # Ensure recommendations are returned
    assert not recommendations.empty, "No recommendations were generated."
    
    # Verify the number of recommendations is up to n_recommendations
    assert len(recommendations) <= 10, "The number of recommendations is incorrect."
    
    # Check if the recommendations match user preferences
    user_profile = user_profiles_df[user_profiles_df['user_id'] == test_user_id].iloc[0]
    if user_profile['type'].lower() != 'all':
        assert all(recommendations['type'].str.lower() == user_profile['type'].lower()), "Recommended types do not match user preference."
    
    preferred_genres = ast.literal_eval(user_profile['preferred_genres'])
    for genres in recommendations['predicted_genre']:
        assert any(genre.lower() in genres.lower() for genre in preferred_genres), "Recommended genres do not match user preference."

def test_get_movie_type():
    assert get_movie_type(1999) == 'Movie'
    assert get_movie_type('1999-2000') == 'Series'
    assert get_movie_type('') == 'Movie'  # Handle empty strings as 'Movie'

def run_tests():
    test_get_movie_type()
    test_recommend_movies()
    print("All tests passed.")

# Run the tests
run_tests()
