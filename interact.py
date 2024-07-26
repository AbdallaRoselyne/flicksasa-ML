import pandas as pd
import random
import boto3
from io import StringIO

# Initialize the S3 client
s3 = boto3.client('s3')
s3_bucket_name = 'myflicksasa'
user_profiles_file_path = 'imdb/simulated_user_profiles.csv'
movies_file_path = 'imdb/combined_processed.csv'
interaction_data_file_path = 'imdb/simulated_user_interactions.csv'

# Function to read CSV from S3
def read_csv_from_s3(bucket_name, file_path):
    obj = s3.get_object(Bucket=bucket_name, Key=file_path)
    return pd.read_csv(obj['Body'])

# Load user profiles from S3
user_profiles_df = read_csv_from_s3(s3_bucket_name, user_profiles_file_path)

# Load movie data from S3
movies_df = read_csv_from_s3(s3_bucket_name, movies_file_path)

# Ensure movie dataset has a unique identifier for each movie
movies_df['movieId'] = range(1, len(movies_df) + 1)

# Function to generate a user interaction
def generate_user_interaction(user_id, movie_id):
    return {
        "userId": user_id,
        "movieId": movie_id,
        "rating": random.randint(1, 5)  # Random rating between 1 and 5
    }

# Generate user interaction data
interaction_data = []
for _, user_profile in user_profiles_df.iterrows():
    user_id = user_profile['user_id']
    for _ in range(random.randint(10, 20)):  # Each user rates between 10 to 20 movies
        movie_id = random.choice(movies_df['movieId'].tolist())
        interaction_data.append(generate_user_interaction(user_id, movie_id))

# Convert to DataFrame
interaction_data_df = pd.DataFrame(interaction_data)

# Save interaction data to CSV in memory
csv_buffer = StringIO()
interaction_data_df.to_csv(csv_buffer, index=False)

# Upload to S3
s3.put_object(Bucket=s3_bucket_name, Key=interaction_data_file_path, Body=csv_buffer.getvalue())

print("Simulated user interaction data has been saved to S3 bucket.")
