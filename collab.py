import pandas as pd
import boto3
from io import StringIO
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import mae, rmse

# Initialize the S3 client
s3_client = boto3.client('s3')

# Function to load CSV data from S3
def load_csv_from_s3(bucket_name, file_key):
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    return pd.read_csv(StringIO(csv_string))

# Load datasets from S3
bucket_name = 'myflicksasa'
movies_df = load_csv_from_s3(bucket_name, 'imdb/combined_processed.csv')
interactions_df = load_csv_from_s3(bucket_name, 'imdb/simulated_user_interactions.csv')

# Add the movieId column to movies_df
movies_df['movieId'] = range(1, len(movies_df) + 1)

# Print the first few rows of the datasets
print("Movies DataFrame:")
print(movies_df.head())
print("Interactions DataFrame:")
print(interactions_df.head())

# Prepare the data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(interactions_df[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train the SVD algorithm
algo = SVD()
algo.fit(trainset)

# Make predictions on the test set
predictions = algo.test(testset)

# Calculate accuracy metrics
print("Mean Absolute Error (MAE):")
mae(predictions)
print("Root Mean Squared Error (RMSE):")
rmse(predictions)

# Function to get movie recommendations for a specific user
def get_recommendations(user_id, num_recommendations=10):
    # Get a list of all movie IDs
    all_movie_ids = movies_df['movieId'].tolist()
    
    # Get the movies the user has already rated
    rated_movies = interactions_df[interactions_df['userId'] == user_id]['movieId'].tolist()
    
    # Get predictions for all movies the user hasn't rated yet
    predictions = [algo.predict(user_id, movie_id) for movie_id in all_movie_ids if movie_id not in rated_movies]
    
    # Sort the predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get the top N recommendations
    top_n_predictions = predictions[:num_recommendations]
    
    # Get the movie details for the top N recommendations
    top_n_movie_ids = [pred.iid for pred in top_n_predictions]
    recommended_movies = movies_df[movies_df['movieId'].isin(top_n_movie_ids)]
    
    return recommended_movies

# Example usage: Get recommendations for a specific user
user_id = 1  # Change this to the user ID you want to get recommendations for
recommendations = get_recommendations(user_id)
print(f"Top 10 movie recommendations for user {user_id}:")
print(recommendations[['title', 'description', 'rating']])
