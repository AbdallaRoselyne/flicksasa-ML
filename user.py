import pandas as pd
import random
import boto3
from io import StringIO

# Define possible values for each preference
moods = ["happy", "sad", "excited", "relaxed"]
types = ["movie", "series"]
genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Fantasy", "Documentary", "Adventure", "Mystery", "Reality-TV"]
ages = range(15, 80)

# Number of simulated users
num_users = 200

# Function to generate a user profile
def generate_user_profile(user_id):
    profile = {
        "user_id": user_id,
        "mood": random.choice(moods),
        "type": random.choice(types),
        "preferred_genres": random.sample(genres, k=random.randint(1, 3)),
        "age": random.choice(ages)
    }
    return profile

# Generate random user profiles
user_profiles = [generate_user_profile(user_id) for user_id in range(1, num_users + 1)]

# Convert to DataFrame
user_profiles_df = pd.DataFrame(user_profiles)

# Save user profiles to CSV
csv_buffer = StringIO()
user_profiles_df.to_csv(csv_buffer, index=False)

# Upload to S3
s3 = boto3.client('s3')
s3_bucket_name = 'myflicksasa'
s3_file_path = 'imdb/simulated_user_profiles.csv'

s3.put_object(Bucket=s3_bucket_name, Key=s3_file_path, Body=csv_buffer.getvalue())

print("User profiles have been saved to S3 bucket.")
